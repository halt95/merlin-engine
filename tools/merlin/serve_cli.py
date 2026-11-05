# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""``merlin``: one front end over the profile registry, preflight and the one launcher.

    merlin serve [checkpoint] [path] [--profile ID | --tp N] [--instances N] [--kv-ram-cache SIZE] [--swarm]
                 [--ple fp8|int6] [--ple-store ram|disk] [--spec mtp|dflash2] [--dry-run] [-- extra vllm args]
    merlin verify <dir> [--checkpoint NAME] [--ple fp8|int6]
    merlin list
    merlin doctor [preflight.py args]
    merlin doctor --report [FILE] [--log FILE] [--lines N] [preflight.py args]

The checkpoint name is optional when the directory can be identified (config.json + the recipes' checkpoint.sha256
lists). Identification does not verify shard bytes: ``merlin verify`` streams all listed files through SHA-256.
Serve only checks the saved size/mtime stamp and prints a hint if it is missing or stale; it never hashes shards.
The profile comes from preflight (the largest TESTED layout that fits this host, else the largest fitting
untested row; a ``testing`` row, run on cloud-hosted hardware but not
parity-qualified, is picked exactly like an untested one), from ``--profile`` or
from ``--tp``; a host with more cards than the layout uses gets a card subset
(P2P-connected where the profile needs it, the tightest topology group first) and a hint for the free cards;
``--instances N`` partitions the host into N disjoint sets across every card class and starts one server per set.
``GPUS`` / ``CUDA_VISIBLE_DEVICES`` always win. Every profile starts
through ``tools/merlin/launch.py``, so a single launch's env + argv equal what the profile's shim produces (instances add
their own ports and caches, see ``plan_instances``). Standard library only.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402
import preflight as pf  # noqa: E402
import registry as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLACEHOLDER = "<checkpoint-dir>"
LAUNCHER = "tools/merlin/launch.py"
# name -> registry key (launch.CHECKPOINTS is the source)
CHECKPOINTS: dict[str, str] = {k: v[0] for k, v in L.CHECKPOINTS.items()}
# the launcher files that pin a profile exactly (every other row starts through merlin serve --profile)
SHIMS: dict[tuple[str, str], str] = {
    ("flash-next", R.REFERENCE_ID): "models/flash-next/serve.sh",
    ("27b", "27b-tp1-3090"): "models/qwen38-27b/serve-1card.sh",
    ("27b", "27b-tp2-3090"): "models/qwen38-27b/serve.sh",
    ("27b", "27b-tp4-3090"): "models/qwen38-27b/serve-tp4.sh",
    ("27b-thinkingcap", "27b-tp2-3090"): "models/thinkingcap/serve.sh",
    ("27b-thinkingcap", "27b-tp4-3090"): "models/thinkingcap/serve-tp4.sh",
    ("nemotron-super-vl", "svl-quad-3090"): "models/nemotron-super-vl/serve.sh",
    ("ornith1.5", "ornith-quad-3090"): "models/ornith1.5/serve.sh",
}
# Flash-Next host RAM per instance (preflight.py RAM_FLOOR_GIB / RAM_WARN_GIB): ~47.7 GiB pinned PLE table + workers,
# plus the KV host tier (MERLIN_KV_HOST_TIER_GIB per rank) when on: the gates read it off ``inv.kv_tier_gib``.
FN_RAM_FLOOR_GIB = pf.RAM_FLOOR_GIB
FN_RAM_WARN_GIB = pf.RAM_WARN_GIB
MIXED_FN_EVIDENCE = (
    "community-validated for mixed RTX 3090 + RTX 3090 Ti quads on the Flash-Next family: one public report "
    "(https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/discussions/1, 2 x 3090 + 2 x 3090 Ti, PCIe 3.0, the v2.5.1 "
    "engine with Flash-Next / Swift) ran without errors; not measured on this engine"
)


NO_P2P_THROUGHPUT_NOTE = (
    "P2P is off on GPUs {gpus}: TP{tp} is the largest-pool layout; for aggregate throughput "
    "without P2P prefer {n} single-card servers (measured on 4 x CMP 170HX 64 GB without P2P, 8 agents x 16K: "
    "TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371; 4 x RTX 4090 showed the same order for prefill): {cmd}"
)


class CliError(Exception):
    def __init__(self, msg: str, code: int = 2):
        super().__init__(msg)
        self.code = code


@dataclass
class Plan:
    checkpoint: str
    profile_id: str
    status: str
    argv: list[str]  # the launcher invocation
    env: dict[str, str]  # what merlin serve adds to the environment
    launcher: str  # repo-relative: always tools/merlin/launch.py
    warnings: list[str] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)
    free: list[int] = field(default_factory=list)  # same-class cards left unused
    skipped: list[int] = field(default_factory=list)  # cards of another class
    notes: list[str] = field(default_factory=list)
    vllm: L.Plan | None = None  # the resolved launcher plan (pure build), for dry-run display


def _profiles_for(name: str, reg: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "flash-next":
        return [p for p in reg["profiles"] if "checkpoint" not in p]
    if name in reg.get("checkpoints", {}):  # a Flash-Next checkpoint row (Swift1.5, Huihui) serves its own rows
        return [p for p in reg["profiles"] if p.get("checkpoint") == name]
    return list(reg["profiles"])


def _load(name: str) -> dict[str, Any]:
    if name not in CHECKPOINTS:
        raise CliError(f"unknown checkpoint {name!r}; known: {', '.join(CHECKPOINTS)}")
    return R.load_registry(CHECKPOINTS[name])


def _gpu_selection(environ: dict[str, str]) -> list[int] | None:
    """One ordered GPU selection from CUDA_VISIBLE_DEVICES / GPUS; disagreeing selectors are refused."""
    sel: dict[str, list[int]] = {}
    for k in ("CUDA_VISIBLE_DEVICES", "GPUS"):
        v = environ.get(k, "").strip()
        if v:
            try:
                sel[k] = [int(x) for x in v.split(",")]
            except ValueError:
                raise CliError(f"{k}={v!r}: only comma-separated GPU indices are supported") from None
    if len(sel) == 2 and sel["CUDA_VISIBLE_DEVICES"] != sel["GPUS"]:
        raise CliError(
            f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']} and GPUS={environ['GPUS']} disagree; set one"
        )
    for k, v in sel.items():
        if len(set(v)) != len(v):
            raise CliError(f"{k}={environ[k]}: a GPU index is given twice")
    return next(iter(sel.values()), None)  # the caller's order is kept: it is the rank / pipeline-stage order


# --------------------------------------------------------------------------------------------
# Card subsets: classes, P2P cliques, tested-name preference, partitions
# --------------------------------------------------------------------------------------------
def _class_of(g: pf.Gpu) -> tuple[str | None, int | None]:
    return g.arch, R.vram_class(pf._gib(g.vram_mib))


def _groups(inv: pf.Inventory) -> list[list[pf.Gpu]]:
    """Homogeneous card groups (arch + VRAM class), largest first, index order inside a group."""
    by: dict[tuple[str | None, int | None], list[pf.Gpu]] = {}
    for g in inv.gpus:
        by.setdefault(_class_of(g), []).append(g)
    return sorted(by.values(), key=lambda gs: (-len(gs), gs[0].index))


def _tested_name(prof: dict[str, Any], g: pf.Gpu) -> bool:
    import re

    pats = prof["match"].get("tested_gpu_names")
    return not pats or any(re.search(p, g.name) for p in pats)


def _clique(inv: pf.Inventory, cards: list[int]) -> bool:
    return inv.p2p_among(cards) == "on"


def _peer_test(inv: pf.Inventory) -> Callable[[list[int]], bool]:
    """``_clique`` with the driver's pair set built once (the partition search tests thousands of sets). A card with
    BAR1 below 1 GiB has no usable peer (NCCL hangs at warmup on any P2P pair that includes it; preflight treats the set
    as P2P off), so such cards never join a P2P-connected set here either."""
    small = {g.index for g in inv.gpus if g.bar1_mib is not None and g.bar1_mib < pf.BAR1_MIN_MIB}
    if inv.p2p_ok is None:
        return lambda cards: len(cards) < 2 or (inv.p2p == "on" and not small & set(cards))
    ok = {(a, b) for a, b in inv.p2p_ok if a not in small and b not in small}
    ok |= {(b, a) for a, b in ok}
    return lambda cards: all((a, b) in ok for a, b in combinations(cards, 2))


def _idle(pool: list[pf.Gpu]) -> list[pf.Gpu]:
    """Cards that hold no other process (more than 1 GiB in use = busy: the pinned pool would not fit beside it)."""
    return [g for g in pool if (g.mem_used_mib or 0) <= pf.BUSY_MIB]


# --------------------------------------------------------------------------------------------
# Topology: how far apart two cards sit (nvidia-smi topo -m links, else the NUMA node, else unknown)
# --------------------------------------------------------------------------------------------
LINK_RANK = {"NV": 0, "PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}
UNKNOWN_RANK = 3  # no topology read: every pair counts as one host bridge apart (index order then decides)


def _link_rank(inv: pf.Inventory, a: pf.Gpu, b: pf.Gpu) -> int:
    """0 NVLink, 1 same PCIe switch, 2 switches apart, 3 same host bridge, 4 same NUMA node, 5 across sockets."""
    kind = inv.link(a.index, b.index)
    if kind is not None:
        return LINK_RANK.get("NV" if kind.upper().startswith("NV") else kind.upper(), UNKNOWN_RANK)
    if a.numa_node is not None and b.numa_node is not None:
        return LINK_RANK["PHB"] if a.numa_node == b.numa_node else LINK_RANK["SYS"]
    return UNKNOWN_RANK


def _set_cost(inv: pf.Inventory, by_index: dict[int, pf.Gpu], cards: list[int]) -> int:
    """The widest link inside one set: what a tensor-parallel group pays on every all-reduce."""
    return max((_link_rank(inv, by_index[a], by_index[b]) for a, b in combinations(cards, 2)), default=0)


def _topo_keys(inv: pf.Inventory, pool: list[pf.Gpu]) -> dict[int, tuple[int, int, int]]:
    """(NUMA node, host bridge group, PCIe switch group) per card index, each group named by its lowest index, so a
    sort over these keys walks the host switch by switch, bridge by bridge, node by node."""
    def groups(limit: int) -> dict[int, int]:
        label = {g.index: g.index for g in pool}
        for a, b in combinations(pool, 2):
            if _link_rank(inv, a, b) <= limit:
                la, lb = label[a.index], label[b.index]
                lo, hi = min(la, lb), max(la, lb)
                for k, v in label.items():
                    if v == hi:
                        label[k] = lo
        return label

    switch, bridge, node = groups(LINK_RANK["PIX"]), groups(LINK_RANK["PHB"]), groups(LINK_RANK["NODE"])
    return {g.index: (g.numa_node if g.numa_node is not None else node[g.index], bridge[g.index], switch[g.index]) for g in pool}


def _order(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int | None = None,
           topo: dict[int, tuple[int, int, int]] | None = None, counts: dict[str, int] | None = None) -> list[pf.Gpu]:
    """The selection order over a card pool: tested-name cards first, then the same model together (the larger model
    count first), then by topology (NUMA node, host bridge, PCIe switch), then index. When fewer than ``need`` cards
    carry a tested name no set can be labelled tested, so the same-model key comes first (2 x 3090 + 4 x 3090 Ti takes
    the homogeneous Ti quad, not a mixed one). With ``need`` given, the tightest complete topology group (PCIe switch,
    then host bridge, then NUMA node) of the leading card's model that holds a whole set moves to the front, so an
    uneven host (NUMA nodes 0,0,1,1,1,1) takes the same-node quad 2-5, not 0-3 across the sockets."""
    if counts is None:  # a partition passes the whole pool's counts and topology labels so every set is ranked alike
        counts = {}
        for g in pool:
            counts[g.name] = counts.get(g.name, 0) + 1
    topo = topo if topo is not None else _topo_keys(inv, pool)
    tested = sum(1 for g in pool if _tested_name(prof, g))
    model_first = need is not None and 0 < tested < need

    def key(g: pf.Gpu) -> tuple:
        t, m = (0 if _tested_name(prof, g) else 1), (-counts[g.name], g.name)
        return (*m, t, *topo[g.index], g.index) if model_first else (t, *m, *topo[g.index], g.index)

    order = sorted(pool, key=key)
    if need and order:
        lead = [g for g in order if g.name == order[0].name]  # the tested-name / same-model keys depend on the name alone
        for depth in (3, 2, 1):  # switch, host bridge, NUMA node: the tightest group holding a whole set leads
            groups: dict[tuple, list[pf.Gpu]] = {}
            for g in lead:
                groups.setdefault(topo[g.index][:depth], []).append(g)
            whole = next((s for s in groups.values() if len(s) >= need), None)
            if whole is not None:
                return whole + [g for g in order if g not in whole]
    return order


def _pick_subset(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int, want_p2p: str,
                 budget: "_Budget | None" = None) -> list[int] | None:
    """``need`` idle cards out of ``pool`` for ``prof`` in ``_order`` (tested names, same model, topology, index):
    P2P-connected when the profile needs it (``want_p2p`` on), preferred when it may use it (any); None when no such
    set exists."""
    idx = [g.index for g in _order(inv, prof, _idle(pool), need)]
    if need > len(idx):
        return None
    if want_p2p in ("on", "any"):
        clique = _peer_test(inv)
        try:  # the first P2P-connected set in order, pruned to each card's peers and bounded (32 cards, no P2P: 32 steps)
            found = _disjoint_cliques(inv, idx, need, 1, clique, budget)
        except _SearchBudget:
            found = None
        if found:
            return found[0]
        if want_p2p == "on":
            return None
    return sorted(idx[:need])


def _partition(inv: pf.Inventory, prof: dict[str, Any], pool: list[pf.Gpu], need: int, n: int, want_p2p: str,
               budget: "_Budget | None" = None) -> list[list[int]] | None:
    """``n`` disjoint sets of ``need`` idle cards, cut one set at a time over ``_order`` of what is left (tested names
    together, same model together, then NUMA node / host bridge / PCIe switch, then index; the same-model key leads
    when the cards left hold no all-tested set), so every set stays homogeneous and inside the tightest topology
    group the counts allow. The 4-card pair shape keeps the reference host's measured stride cut (0,2 / 1,3) unless
    the topology says the contiguous cut is tighter. When the cut does not satisfy the P2P need, a backtracking search
    over P2P-connected sets of the whole pool follows (None = no such partition exists; CliError when the search runs
    past its budget and the row needs P2P, which is not proof that none exists)."""
    pool = _idle(pool)
    if n * need > len(pool):
        return None
    tested = [g for g in pool if _tested_name(prof, g)]
    mixed = 0 < len(tested) < len(pool)
    left = list(pool)
    topo = _topo_keys(inv, pool)
    counts: dict[str, int] = {}
    for g in pool:
        counts[g.name] = counts.get(g.name, 0) + 1
    contiguous: list[list[int]] = []
    for _ in range(n):  # re-ordered after every complete set: 6 x 3090 + 8 x 3090 Ti -> 3090 quad, Ti quad, Ti quad
        take = _order(inv, prof, left, need, topo, counts)[:need]
        contiguous.append(sorted(g.index for g in take))
        left = [g for g in left if g not in take]
    head = [c for s in contiguous for c in s]
    idx = head + [g.index for g in _order(inv, prof, left, None, topo, counts)]  # the whole pool: the search may need more cards
    cuts = [contiguous]
    if len(pool) == 4 and need == 2 and n == 2 and not mixed:  # the reference shape: stride is the measured best
        stride = [sorted(head[i::n]) for i in range(n)]
        by_index = {g.index: g for g in pool}
        if sum(_set_cost(inv, by_index, s) for s in stride) <= sum(_set_cost(inv, by_index, s) for s in contiguous):
            cuts = [stride, contiguous]
        else:
            cuts = [contiguous, stride]
    if want_p2p not in ("on", "any"):
        return cuts[0]
    clique = _peer_test(inv)
    for cut in cuts:
        if all(clique(s) for s in cut):
            return cut
    try:
        sets = _disjoint_cliques(inv, idx, need, n, clique, budget)
    except _SearchBudget:
        if want_p2p == "on":
            raise CliError(
                f"the search for {n} disjoint P2P-connected sets of {need} cards over {len(idx)} cards ran past its budget of "
                f"{PARTITION_BUDGET} set tests (not proof that none exists); start the instances by hand (GPUS=<set> PORT=<port> merlin serve ...)"
            ) from None
        return cuts[0]
    if sets is not None:
        return sets
    return None if want_p2p == "on" else cuts[0]


PARTITION_BUDGET = 100_000  # clique tests: far above any 8-card host, a bound for larger hosts


class _SearchBudget(Exception):
    pass


class _Budget:
    """One search allowance shared by every clique search of one command (a plain serve's hints, one --instances
    plan), so retries over set counts and rows cannot each spend the whole PARTITION_BUDGET."""

    def __init__(self, limit: int | None = None):
        self.tests, self.limit, self.exhausted = 0, limit or PARTITION_BUDGET, False

    def spend(self) -> None:
        self.tests += 1
        if self.tests > self.limit:
            self.exhausted = True
            raise _SearchBudget()


def _disjoint_cliques(
    inv: pf.Inventory, idx: list[int], need: int, n: int, clique: Callable[[list[int]], bool] | None = None,
    budget: _Budget | None = None,
) -> list[list[int]] | None:
    """The first ``n`` disjoint P2P-connected sets of ``need`` cards in ``idx`` order, by backtracking: the first
    remaining card joins the next set (with cards it has peer access to) or stays unused. Cards with fewer than
    ``need - 1`` peers in ``idx`` can belong to no set and are dropped first; fewer than ``need x n`` cards left is a
    proven None without a search. ``_SearchBudget`` when the search exceeds the budget's clique tests."""
    clique = clique or _peer_test(inv)
    budget = budget or _Budget()
    if budget.exhausted:
        raise _SearchBudget()
    if need > 1:
        peers_of = {c: [d for d in idx if d != c and clique([c, d])] for c in idx}
        kept = [c for c in idx if len(peers_of[c]) >= need - 1]
    else:
        kept = list(idx)
    if len(kept) < need * n:
        return None

    def search(left: list[int], k: int) -> list[list[int]] | None:
        if k == 0:
            return []
        if len(left) < need * k:
            return None
        first, rest = left[0], left[1:]
        peers = [c for c in rest if clique([first, c])]
        if len(peers) >= need - 1:
            for others in combinations(peers, need - 1):
                budget.spend()
                s = [first, *others]
                if clique(s):
                    sub = search([i for i in rest if i not in others], k - 1)
                    if sub is not None:
                        return [sorted(s), *sub]
        return search(rest, k)

    return search(kept, n)


def _no_p2p_twin(p: dict[str, Any]) -> bool:
    """A 27B row for hosts without peer-to-peer (match p2p "off"): the untested no-P2P twin of a tested 3090 row, same
    layout and pin. It is ranked with the tested rows, so the 27B default on a no-P2P host stays the largest layout
    rather than falling back to the tested one-card row. Flash-Next's no-P2P rows keep their own order (unchanged until
    measured on cards without P2P)."""
    return p["status"] != "tested" and p["match"].get("p2p") == "off" and p["id"].startswith("27b-")


def _ordered_candidates(reg: dict[str, Any], name: str, profile: str | None, tp: int | None) -> list[dict[str, Any]]:
    """Selection order: tested multi-card rows (most cards first), then the 27B no-P2P twins (most cards first), then the
    tested single-card rows, then the rest; rank (the registry's own preference) breaks ties."""
    cands = _profiles_for(name, reg)
    if profile:
        cands = [p for p in cands if p["id"] == profile]
    if tp is not None:
        cands = [p for p in cands if (p.get("layout") or {}).get("tp", p["serve"].get("tensor_parallel")) == tp]
    def tier(p: dict[str, Any]) -> int:
        if not profile and not R.auto_selectable(p):
            return 4  # a named alternative (auto_select false): last
        if p["status"] == "tested":
            return 0 if p["match"]["cards"] > 1 else 2
        return 1 if _no_p2p_twin(p) else 3
    return sorted(cands, key=lambda p: (tier(p), -p["match"]["cards"], p.get("rank", 9)))


def _fits_class(p: dict[str, Any], group: list[pf.Gpu]) -> bool:
    arch, vclass = _class_of(group[0])
    return arch in p["match"]["arch"] and p["match"]["vram_gib"] == vclass


def _want_p2p(reg: dict[str, Any], p: dict[str, Any], inv: pf.Inventory, group: list[pf.Gpu]) -> str:
    """What the profile needs from the subset: on (must be connected), any (prefer), off (no requirement)."""
    m = p["match"]["p2p"]
    if reg["model"] in pf.STRICT_P2P_MODELS:
        return m  # on / off / any as the row declares
    return "on" if m == "on" else "any"


@dataclass
class Choice:
    rep: pf.Report
    prof: dict[str, Any]
    cards: list[int]
    free: list[int]
    skipped: list[int]


def choose(
    reg: dict[str, Any], name: str, inv: pf.Inventory, profile: str | None, tp: int | None, want_cards: int | None = None,
    groups: list[list[pf.Gpu]] | None = None,
) -> Choice:
    """Pick a profile and a card set for ``name`` on ``inv``: the largest TESTED layout that fits a homogeneous card
    group, else the largest fitting untested row; smaller layouts take a subset. ``groups`` restricts the card groups
    considered (``--instances`` plans one class at a time). Raises CliError when nothing fits."""
    groups = _groups(inv) if groups is None else groups
    reasons: list[str] = []
    for status_pass in ("tested", "untested"):
        for group in groups:
            for p in _ordered_candidates(reg, name, profile, tp):
                if status_pass == "tested" and p["status"] != "tested" and not _no_p2p_twin(p):
                    continue  # the untested pass retries tested rows too: on other card names they run untested
                named = not profile and not R.auto_selectable(p)
                if status_pass == "tested" and named:
                    continue  # a named alternative waits for the last pass
                if not _fits_class(p, group):
                    continue
                need = p["match"]["cards"]
                if want_cards is not None and need != want_cards:
                    continue
                cls = f"{'/'.join(p['match']['arch'])} {p['match']['vram_gib']} GB"
                if need > len(group):
                    reasons.append(f"{p['id']}: needs {need} {cls} cards; this host has {len(group)} of that class")
                    continue
                if need > len(_idle(group)):
                    busy = [g.index for g in group if g not in _idle(group)]
                    reasons.append(f"{p['id']}: needs {need} idle {cls} cards; GPUs {','.join(map(str, busy))} already hold more than "
                                   f"1 GiB (skipped; GPUS={','.join(str(g.index) for g in group[:need])} forces them)")
                    continue
                cards = _pick_subset(inv, p, group, need, _want_p2p(reg, p, inv, group))
                if cards is None:
                    reasons.append(f"{p['id']}: needs {need} P2P-connected {cls} cards (peer-to-peer is {inv.p2p_among([g.index for g in group])} among them)")
                    continue
                if not profile and _no_p2p_twin(p) and inv.p2p_among(cards) == "on":
                    continue  # automatic selection: cards with peer-to-peer run the P2P row of the same layout
                rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=p["id"], gpus=cards)
                if rep.exit_code != 0 or not rep.profile_id:
                    reasons += [f"{p['id']}: {f.text}" for f in rep.findings if f.level == "FAIL"]
                    continue
                if status_pass == "tested" and rep.status != "tested" and not _no_p2p_twin(p):
                    continue  # a tested row counts as tested only on an all-tested-name set; try it again as untested
                free = [g.index for g in group if g.index not in cards]
                skipped = [g.index for g in inv.gpus if g.index not in cards and g.index not in free]
                return Choice(rep, p, cards, free, skipped)
    if profile and not reasons:
        want = R.get_profile(reg, profile)["match"]
        have = sorted({f"{g.arch} {R.vram_class(pf._gib(g.vram_mib))} GB" for g in inv.gpus})
        reasons.append(f"--profile {profile} does not match this hardware: it needs {want['cards']} x "
                       f"{'/'.join(want['arch'])} {want['vram_gib']} GB; found {', '.join(have)}")
    why = pf.evaluate(inv, reg, model=reg["model"]).no_profile_reason if not profile and tp is None else None
    if why and not why.startswith("mixed GPUs"):  # the whole-host explanation; a pinned row's refusal is its own line above
        reasons.append(why)
    reasons.append(f"run `merlin doctor --model {reg['model']}` for the full report")
    raise CliError(
        f"no usable {name} profile on this hardware.\n  " + "\n  ".join(dict.fromkeys(reasons)), 3
    )


# --------------------------------------------------------------------------------------------
# One plan
# --------------------------------------------------------------------------------------------
def _hint_cmd(name: str, ckpt: str, free: list[int], port: int, tp: int | None = None, profile: str | None = None,
              instances: int | None = None) -> str:
    """A merlin serve command for the hints: ``GPUS=.. PORT=..`` when cards / a port are given, ``--tp`` / ``--profile``
    carried over, ``--instances N`` when asked."""
    pre = (f"GPUS={','.join(map(str, free))} " if free else "") + (f"PORT={port} " if port else "")
    tail = (f" --tp {tp}" if tp else "") + (f" --profile {profile}" if profile else "") + (f" --instances {instances}" if instances else "")
    return f"{pre}merlin serve {name} {shlex.quote(ckpt) if ckpt != PLACEHOLDER else ckpt}{tail}"


def _instance_env_hint(repo: Path, i: int, port: int) -> str:
    """What ``--instances`` would export for instance ``i`` on ``port`` beyond GPUS / PORT: its own cache root (every
    builder maps CACHE_ROOT to VLLM_CACHE_ROOT + TRITON_CACHE_DIR) and the distributed ports derived from PORT, so a
    hand-started second server shares nothing with instance 0."""
    m, d = L.distributed_ports(str(port))
    return f"CACHE_ROOT={shlex.quote(f'{repo.as_posix()}/.vllm-cache/instance-{i}')} MASTER_PORT={m} VLLM_PORT={d}"


def _tp1_fits(reg: dict[str, Any], inv: pf.Inventory, group: list[pf.Gpu], card: int) -> bool:
    """Whether some single-card row of this class passes preflight on ``card`` (A10G / L4 pins fall under the floor)."""
    return any(pf.evaluate(inv, reg, model=reg["model"], want_profile=q["id"], gpus=[card]).exit_code == 0
               for q in _profiles_for_model(reg) if q["match"]["cards"] == 1 and _fits_class(q, group))


def _profiles_for_model(reg: dict[str, Any]) -> list[dict[str, Any]]:
    return list(reg["profiles"])


def _throughput_note(reg: dict[str, Any], name: str, ckpt: str, inv: pf.Inventory, p: dict[str, Any], cards: list[int],
                     n_class: int) -> str:
    """Item 5's NOTE: the measurements always, the ``--tp 1 --instances N`` command only when a single-card row of this
    class passes preflight on these cards."""
    group = [g for g in inv.gpus if _class_of(g) == _class_of(next(x for x in inv.gpus if x.index == cards[0]))]
    if _tp1_fits(reg, inv, group, cards[0]):
        cmd = _hint_cmd(name, ckpt, [], 0, tp=1, instances=n_class)
    else:
        cmd = "no single-card 27B row fits these cards (their VRAM leaves no 262K pool on one card), so this layout stands"
    return NO_P2P_THROUGHPUT_NOTE.format(tp=p["match"]["cards"], gpus=",".join(map(str, cards)), n=n_class, cmd=cmd)


def _free_hint(reg: dict[str, Any], name: str, ckpt: str, inv: pf.Inventory, p: dict[str, Any], group: list[pf.Gpu],
               free: list[int], port: int, tp: int | None, profile: str | None, repo: Path) -> str:
    """What the idle cards of the chosen class left over can run: ``--instances k`` when k sets of the chosen layout fit
    the class, and one ``GPUS=.. PORT=.. CACHE_ROOT=..`` command for the largest row of this checkpoint that fits the
    free cards (the chosen one under --tp / --profile); or why nothing fits."""
    need = p["match"]["cards"]
    idle = len(_idle(group))
    budget = _Budget()
    k = 0
    for k in range(idle // need, 1, -1):  # how many sets of the chosen layout the idle cards really hold
        if not _fits_ram(reg, inv, [need] * k):
            continue
        try:
            if _partition(inv, p, group, need, k, _want_p2p(reg, p, inv, group), budget) is not None:
                break
        except CliError:
            continue
    else:
        k = 0
    parts: list[str] = []
    flags = (f" --tp {tp}" if tp else "") + (f" --profile {profile}" if profile else "")
    if k >= 2:
        parts.append(f"--instances {k}{flags} ({k} x {p['id']} on the {idle} idle cards of this class)")
    found = _hint_set(reg, name, inv, p, group, free, tp, profile, beside=[need], budget=budget)
    if not isinstance(found, str):
        q, pick = found
        same = q is p
        parts.append(f"{_instance_env_hint(repo, 1, port)} {_hint_cmd(name, ckpt, pick, port, tp=tp if same else None, profile=profile if same else None)}"
                     + ("" if same else f" ({q['id']}: the {len(free)} free cards hold no {p['id']})"))
        return f"GPUs {','.join(map(str, free))} free: {', or '.join(parts)}"
    why = found
    if parts:
        return f"GPUs {','.join(map(str, free))} free: {parts[0]} (no {name} layout for {len(free)} x {_class_name(group[0])}: {why})"
    return f"GPUs {','.join(map(str, free))} unused: no {name} layout for {len(free)} x {_class_name(group[0])} ({why})"


def _hint_set(reg: dict[str, Any], name: str, inv: pf.Inventory, p: dict[str, Any], group: list[pf.Gpu], free: list[int],
              tp: int | None, profile: str | None, beside: list[int] = (), budget: "_SearchBudget | None" = None,
              ) -> tuple[dict[str, Any], list[int]] | str:
    """The row and cards a hinted second server would take: the chosen row first (under --tp / --profile), then the
    largest row of this checkpoint that fits the free cards with a set that runs: P2P-connected where the row needs
    it, passing preflight on exactly those cards (the same side-effect-free check merlin serve applies to them), and
    fitting the host RAM beside the servers in ``beside`` (card counts). A string says why nothing fits."""
    ranked = [q for q in _ordered_candidates(reg, name, None, None) if _fits_class(q, group)]
    rows = [p, *[q for q in ranked if q is not p]] if (tp is not None or profile) else ranked
    pool = [g for g in group if g.index in free]
    sized = [q for q in rows if q["match"]["cards"] <= len(free)]
    refused: list[str] = []
    for q in sized:
        if not _fits_ram(reg, inv, [*beside, q["match"]["cards"]]):
            refused.append(f"host RAM {_ram_budget_gib(inv):.0f} GiB holds no {q['match']['cards']}-card Flash-Next server beside the "
                           f"one started (~{pf.ram_floor_gib(q['match']['cards'], inv.kv_tier_gib, inv.ple_table_gib):.0f} GiB more"
                           f"{pf.ram_tier_text(q['match']['cards'], inv.kv_tier_gib)})")
            continue
        pick = _pick_subset(inv, q, pool, q["match"]["cards"], _want_p2p(reg, q, inv, group), budget)
        if not pick:
            continue
        rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=q["id"], gpus=pick)
        if rep.exit_code == 0 and rep.profile_id:
            return q, pick
        refused += [f"{q['id']} on GPUs {','.join(map(str, pick))}: {f.text}" for f in rep.findings if f.level == "FAIL"]
    if refused:
        return "; ".join(dict.fromkeys(refused))
    if sized:
        return f"no P2P-connected set for a {sized[0]['match']['cards']}-card row among them (peer-to-peer is {inv.p2p_among(free)})"
    smallest = min((q["match"]["cards"] for q in _profiles_for(name, reg) if _fits_class(q, group)), default=None)
    return f"the smallest {name} row of this class takes {smallest} card(s)" if smallest else f"no {name} row of this class"


def _ram_budget_gib(inv: pf.Inventory) -> float | None:
    """The host RAM new Flash-Next servers can share: what is available now on a probed host (running servers are
    already out of it), else MemTotal / the cgroup limit; None when unknown."""
    ram = inv.host.ram_mib
    if inv.source == "probe":
        avail = _mem_available_mib()
        if avail is not None and (ram is None or avail < ram):
            ram = avail
    return None if ram is None else pf._gib(ram)


def _fits_ram(reg: dict[str, Any], inv: pf.Inventory, sizes: list[int]) -> bool:
    """Whether Flash-Next servers of these card counts fit the host RAM together (other families pin no host table)."""
    if reg["model"] != "flash-next":
        return True
    budget = _ram_budget_gib(inv)
    return budget is None or sum(pf.ram_floor_gib(s, inv.kv_tier_gib, inv.ple_table_gib) for s in sizes) <= budget


def _default_port(reg: dict[str, Any]) -> int:
    return int(reg.get("launch", {}).get("port") or reg["base"].get("port", 8000))


def resolve(
    name: str,
    path: str | None,
    profile: str | None,
    inv: pf.Inventory,
    extra: list[str],
    repo: Path = REPO,
    environ: dict[str, str] | None = None,
    tp: int | None = None,
) -> Plan:
    env_in = os.environ if environ is None else environ
    try:
        reg = R.with_kv_dtype(_load(name), env_in.get("KV_DTYPE"))
    except R.ProfileError as e:
        raise CliError(str(e)) from None
    ckpt = path or env_in.get("MERLIN_CHECKPOINT_DIR") or PLACEHOLDER
    # the RAM gates count the KV host tier this launch asks for and the PLE table this checkpoint carries (INT6 g32)
    inv = replace(inv, kv_tier_gib=pf.kv_host_tier_gib(env_in), ple_table_gib=pf.ple_table_gib_of(ckpt, inv.ple_table_gib, env_in.get("PLE_STORE")))
    if not (repo / LAUNCHER).is_file():
        raise CliError(f"{LAUNCHER} is not in this tree ({repo}); nothing started.")
    sel = _gpu_selection(env_in)
    ids = [p["id"] for p in _profiles_for(name, reg)]
    if profile and profile not in ids:
        raise CliError(f"--profile {profile!r} does not belong to {name}; its profiles: {', '.join(ids)}")
    notes: list[str] = []
    if sel is not None:
        # GPUS / CUDA_VISIBLE_DEVICES: exactly those cards (preflight checks the set it is given)
        picked = [g for g in inv.gpus if g.index in sel]
        cands = [R.get_profile(reg, profile)] if profile else _ordered_candidates(reg, name, None, tp)
        if tp is not None and picked and len({_class_of(g) for g in picked}) == 1:
            shaped = [q for q in cands if _fits_class(q, picked) and q["match"]["cards"] == len(picked)]
            if not shaped:
                sizes = sorted({q["match"]["cards"] for q in cands if _fits_class(q, picked)})
                raise CliError(
                    f"--tp {tp} on {_class_name(picked[0])} takes {' or '.join(map(str, sizes)) or 'no'} card(s); "
                    f"GPUS={','.join(map(str, sel))} names {len(picked)}")
            cands = shaped
        order = [q["id"] for q in cands]
        rep = None
        if profile or tp is not None or reg["model"] == "flash-next" and name != "flash-next":
            for pid in order:
                rep = pf.evaluate(inv, reg, model=reg["model"], want_profile=pid, gpus=sel)
                if rep.exit_code == 0:
                    break
        else:
            rep = pf.evaluate(inv, reg, model=reg["model"], gpus=sel)
            if rep.exit_code == 0 and rep.profile_id not in ids:
                raise CliError(f"auto-selected {rep.profile_id} is not a {name} profile; pass --profile")
        if rep is None or rep.exit_code != 0 or not rep.profile_id:
            fails = [f"{f.code}: {f.text}" for f in (rep.findings if rep else []) if f.level == "FAIL"]
            raise CliError(
                f"no usable {name} profile on this hardware.\n  " + "\n  ".join(fails or ["no profile matched"]),
                rep.exit_code if rep and rep.exit_code else 3,
            )
        p = R.get_profile(reg, rep.profile_id)
        cards, free, skipped = [g.index for g in rep.inventory.gpus], [], []
    else:
        ch = choose(reg, name, inv, profile, tp)
        rep, p, cards, free, skipped = ch.rep, ch.prof, ch.cards, ch.free, ch.skipped
        if skipped:
            notes.append(f"GPUs {','.join(map(str, skipped))} skipped: another GPU class than the chosen set (a layout never spans classes; "
                         f"--instances N fills every class that has a row of the layout)")
        group = [g for g in inv.gpus if g.index in cards + free]
        busy_free = [g.index for g in group if g.index in free and g not in _idle(group)]
        free = [i for i in free if i not in busy_free]
        if busy_free:
            notes.append(f"GPUs {','.join(map(str, busy_free))} busy: they hold more than 1 GiB and are not offered to another server")
        if free:
            # The larger row replaces the chosen server, so budget its RAM alone.
            larger = next((q for q in _ordered_candidates(reg, name, None, None)
                           if _fits_class(q, group) and q["match"]["cards"] >= 8
                           and len(cards) < q["match"]["cards"] <= len(cards) + len(free)
                           and _fits_ram(reg, inv, [q["match"]["cards"]])), None)
            if larger is not None:
                notes.append(f"{len(free)} matching cards stay idle with {p['id']}; select the "
                             f"{larger['match']['cards']}-card row ({R.label_text(reg, larger)}) "
                             f"with --profile {larger['id']}")
            notes.append(_free_hint(reg, name, ckpt, inv, p, group, free, int(env_in.get("PORT") or _default_port(reg)) + 1,
                                    tp, profile, repo))
    names = sorted({g.name for g in rep.inventory.gpus})
    status = rep.status or p["status"]
    if reg["model"] == "27b" and len(cards) > 1 and rep.effective_p2p != "on":
        n_class = len(cards) if sel is not None else len(cards) + len(_idle([g for g in inv.gpus if g.index in free]))
        notes.append(_throughput_note(reg, name, ckpt, inv, p, cards, n_class))
    if len(names) > 1:
        notes.append(f"mixed cards in one layout ({' + '.join(names)}): it runs at the slower card's pace; P2P {rep.effective_p2p} across the set")
        if reg["model"] == "flash-next" and status == "untested" and all("3090" in n for n in names):
            status = "untested"
            notes.append(MIXED_FN_EVIDENCE)
    env: dict[str, str] = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))}
    model = reg["model"]
    if "KV_DTYPE" in env_in:
        env["KV_DTYPE"] = env_in["KV_DTYPE"]
    busy = [g.index for g in rep.inventory.gpus if (g.mem_used_mib or 0) > pf.BUSY_MIB]
    if busy:  # only reachable through GPUS / CUDA_VISIBLE_DEVICES: the selector skips busy cards
        notes.append(f"GPUs {','.join(map(str, busy))} already hold more than 1 GiB (another process?); the pinned pool may not fit")
    if model == "flash-next" and inv.source == "probe":
        avail = _mem_available_mib()
        if avail is not None and pf._gib(avail) < pf.ram_floor_gib(len(cards), inv.kv_tier_gib, inv.ple_table_gib):
            raise CliError(pf.ram_available_text(len(cards), avail, inv.kv_tier_gib, inv.ple_table_gib) + ". Nothing started.")
    if model in ("27b", "svl"):
        if p["match"]["cards"] > 1 and p["match"]["p2p"] == "on" and rep.effective_p2p != "on":
            raise CliError(
                f"{p['id']} was qualified with peer-to-peer and P2P is {rep.effective_p2p} on these cards; refusing: "
                + pf.no_p2p_hint(reg, model, p["match"]["arch"][0], p["match"]["vram_gib"], cards, rep.effective_p2p, inv)
            )
        if p["match"]["cards"] > 1 and p["match"]["p2p"] == "any":
            env["P2P"] = rep.effective_p2p  # the launcher takes P2P=on|off
        elif p["match"]["cards"] > 1 and p["match"]["p2p"] == "off":
            env["P2P"] = "off"  # a no-P2P row runs without peer-to-peer whatever the driver reports
        env["GPUS"] = env["CUDA_VISIBLE_DEVICES"]
    elif model == "ornith":
        env["GPUS"] = env["CUDA_VISIBLE_DEVICES"]
        if p["match"]["cards"] > 1:
            env["P2P"] = p["match"]["p2p"]  # the row was qualified on exactly this interconnect
    else:
        if rep.p2p_mode and (R.auto_p2p_switch(p) or rep.p2p_mode == "ipc"):
            env["P2P"] = rep.p2p_mode
    if rep.adjusted is not None:  # the pin preflight derived for these cards (whole blocks where the row records them)
        pin = rep.adjusted["pin_bytes"]
        dfl = p["serve"].get("dflash2") if model == "flash-next" and (env_in.get("SPEC") or "mtp") == "dflash2" else None
        if dfl:  # the derivation is from the MTP pin: lower the DFlash2 pin by the same bytes, in its whole blocks
            lowered = p["serve"]["kv_cache_memory"] - pin
            bpb = dfl["kv_cache_memory"] // dfl["kv_blocks"]
            pin = max(0, (dfl["kv_cache_memory"] - lowered) // bpb * bpb)
            notes.append(f"SPEC=dflash2: the DFlash2 KV pin is lowered by the same {lowered:,} B for these cards, in whole "
                         f"blocks: KV_BYTES={pin} (in testing: DFlash2 is validated at the qualified pin only)")
        env["KV_BYTES"] = str(pin)
        maxlen = reg.get("launch", {}).get("max_model_len")
        if model == "svl" and maxlen and rep.adjusted["pool_tokens"] < maxlen:
            # Super VL serves its native 1M by default; a reduced pool that cannot hold one 1M session serves 262K
            env["MAXLEN"] = "262144"
            notes.append(f"the reduced KV pool (~{rep.adjusted['pool_tokens']:,} tokens) holds no {maxlen:,}-token session on these "
                         f"cards: MAXLEN=262144")
    for k in ("KV_BYTES", "P2P", "MAXLEN"):  # an explicit caller value always wins over what preflight derived
        if k in env and env_in.get(k) and env_in[k] != env[k]:
            notes.append(f"{k}={env_in[k]} from your environment overrides the derived {k}={env[k]}")
            env[k] = env_in[k]
    argv = [sys.executable, str(repo / LAUNCHER), "--model", model, "--profile", p["id"], "--checkpoint", name, "--status", status,
            "--", ckpt, *extra]
    # The launcher prints the sm_80 FP8 warning once when it starts.
    warns = [f"{f.code}: {f.text}" for f in rep.findings if f.level == "WARN" and f.code != "kv-dtype"]
    try:
        vllm = L.build(model, p["id"], ckpt, {**env_in, **env}, extra, None, name, repo, status=status)
    except L.LaunchError as e:
        raise CliError(f"{p['id']}: {e}", e.code) from None
    if model == "flash-next":
        warns += L.dflash2_override_warnings(vllm.profile, vllm.caller_env)
    return Plan(name, p["id"], status, argv, env, LAUNCHER, warns, cards, free, skipped, notes, vllm)


def missing_resources(plan: Plan, environ: dict[str, str]) -> list[str]:
    """Launch inputs the launcher will refuse without (checked here so --dry-run can say so without starting)."""
    out: list[str] = []
    v = plan.vllm
    if v is None:
        return out
    if v.family == "flash-next":
        tree = v.tree or str(v.repo)
        if not (Path(tree) / "vllm").is_dir():
            out.append(f"TREE={tree} (no vllm/ package)")
    for path, msg in v.files:
        if not Path(path).is_file():
            out.append(msg)
    return out


# --------------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------------
def _running_fn_instances() -> int:
    """Flash-Next servers already running on this host (their environment carries the PLE offload timeout)."""
    n = 0
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    envb = f.read()
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read()
            except OSError:
                continue
            if b"VLLM_PLE_OFFLOAD_READY_TIMEOUT=" in envb and b"vllm" in cmd and b"serve" in cmd and b"VLLM_PP_LAYER_PARTITION" in envb:
                n += 1
    except OSError:
        return 0
    return n  # every rank of one server counts; divide by the layout size below


def _class_name(g: pf.Gpu) -> str:
    return f"{g.arch} {R.vram_class(pf._gib(g.vram_mib))} GB"


def _gpus_of(group: list[pf.Gpu]) -> str:
    return f"GPUs {','.join(str(g.index) for g in group)} ({_class_name(group[0])})"


def _allocate_instances(
    reg: dict[str, Any], name: str, inv: pf.Inventory, profile: str | None, tp: int | None, n: int,
) -> tuple[list[tuple[dict[str, Any], list[int]]], list[str]]:
    """``n`` (profile, card set) pairs over every card class of the host, each set homogeneous and running its own
    row. Per layout size (the ``--tp`` / ``--profile`` one, else every size): class by class (tested first, then the
    larger VRAM class, then index), ``choose`` picks the row for the idle cards, its disjoint sets are cut, and
    ``choose`` runs again over the cards left so another row of the same size (the no-P2P quad behind a P2P quad) can
    fill them. Flash-Next sets count against the host RAM as they are added. Among the sizes that reach ``n`` the one
    with the most tested instances wins, then the larger size. Returns the sets and the notes (classes left out and
    why). Raises CliError naming the best attempt (sets, unused cards, what stopped it) when fewer than ``n`` fit."""
    groups = _groups(inv)
    sizes: list[int | None] = [None] if profile or tp is not None else sorted(
        {q["match"]["cards"] for q in _profiles_for(name, reg)}, reverse=True)
    budget = _Budget()
    ram_budget = _ram_budget_gib(inv) if reg["model"] == "flash-next" else None
    attempts: list[tuple[list[tuple[dict[str, Any], list[int]]], int, list[str], list[str], list[tuple[list[pf.Gpu], str]], int]] = []
    for size in sizes:
        sets: list[tuple[dict[str, Any], list[int]]] = []
        tested_n = 0
        have: list[str] = []
        short: list[str] = []
        left_out: list[tuple[list[pf.Gpu], str]] = []  # (class group, why it has no row)
        classes: list[tuple[Choice, list[pf.Gpu]]] = []
        for group in groups:
            try:
                classes.append((choose(reg, name, inv, profile, tp, want_cards=size, groups=[group]), group))
            except CliError as e:
                why = str(e).split("\n")
                left_out.append((group, why[1].strip() if len(why) > 1 else why[0]))
        classes.sort(key=lambda t: (0 if t[0].rep.status == "tested" else 1, -(t[0].prof["match"]["vram_gib"]), t[1][0].index))
        remaining = n
        ram_used = 0.0
        for ch, group in classes:
            pool = _idle(group)
            first = True
            while remaining and pool:
                if not first:
                    try:
                        ch = choose(reg, name, inv, profile, tp, want_cards=size, groups=[pool])
                    except CliError:
                        break
                first = False
                p = ch.prof
                need = p["match"]["cards"]
                want = _want_p2p(reg, p, inv, group)
                k_max = min(len(pool) // need, remaining)
                if ram_budget is not None:
                    each = pf.ram_floor_gib(need, inv.kv_tier_gib, inv.ple_table_gib)
                    k_ram = int((ram_budget - ram_used) // each)
                    if k_ram < k_max:
                        short.append(f"host RAM {ram_budget:.0f} GiB holds {int(ram_budget // each)} x {need}-card "
                                     f"Flash-Next server(s) (~{each:.0f} GiB each{pf.ram_tier_text(need, inv.kv_tier_gib)})")
                        k_max = k_ram
                if k_max < 1:
                    break
                got: list[list[int]] = []
                budget_hit = False
                for k in range(k_max, 0, -1):
                    try:
                        found = _partition(inv, p, pool, need, k, want, budget)
                    except CliError as e:
                        short.append(f"{_gpus_of(pool)}: {e}")
                        budget_hit = True
                        break  # the budget is spent: smaller k would raise at once too
                    if found is not None:
                        got = found
                        break
                if not got:
                    if not budget_hit:
                        short.append(f"{_gpus_of(pool)}: no disjoint P2P-connected set of {need} cards for {p['id']} "
                                     f"(peer-to-peer is {inv.p2p_among([g.index for g in pool])} among them)")
                    break
                if len(got) < k_max:
                    short.append(f"{_gpus_of(pool)}: {len(got)} of {k_max} disjoint P2P-connected sets of {need} cards for {p['id']} "
                                 f"(peer-to-peer is {inv.p2p_among([g.index for g in pool])} among them)")
                sets += [(p, cards) for cards in got]
                tested_n += len(got) if (ch.rep.status == "tested" or _no_p2p_twin(p)) else 0
                have.append(f"{len(got)} x {p['id']} on GPUs " + " / ".join(",".join(map(str, c)) for c in got))
                remaining -= len(got)
                ram_used += len(got) * pf.ram_floor_gib(need, inv.kv_tier_gib, inv.ple_table_gib)
                used = {c for cards in got for c in cards}
                pool = [g for g in pool if g.index not in used]
        attempts.append((sets, tested_n, have, short, left_out, size or 0))
    best = max(attempts, key=lambda a: (len(a[0]) >= n, a[1], len(a[0]), a[5]))
    sets, tested_n, have, short, left_out, _size = best
    if len(sets) >= n:
        return sets[:n], [f"{_gpus_of(group)} not used: {why}" for group, why in left_out]
    used = {c for _, cards in sets for c in cards}
    unused = [g.index for group in groups for g in _idle(group) if g.index not in used]
    lines = [f"{_gpus_of(group)}: {why}" for group, why in left_out] + short
    raise CliError(f"--instances {n}: this host holds {len(sets)} instance(s): {'; '.join(have) or 'none'}"
                   + (f"; unused GPUs {','.join(map(str, unused))}" if unused else "")
                   + "".join(f"\n  {ln}" for ln in dict.fromkeys(lines)))


def plan_instances(
    name: str, path: str | None, profile: str | None, inv: pf.Inventory, extra: list[str], n: int,
    repo: Path = REPO, environ: dict[str, str] | None = None, tp: int | None = None, served_suffix: bool = False,
) -> list[Plan]:
    """``n`` disjoint card sets across the host's card classes (``_allocate_instances``: each set homogeneous, each class
    on its own row), each resolved as ``GPUS=<set> PORT=<base+i> merlin serve`` plus what an instance needs of its own:
    ``MASTER_PORT`` / ``VLLM_PORT`` from ``launch.distributed_ports(PORT)`` (exported on every family; a single
    Flash-Next or 27B launch exports none) and, after the first instance, ``CACHE_ROOT``, ``VLLM_CACHE_ROOT`` and
    ``TRITON_CACHE_DIR`` under ``<cache root>/instance-<i>``."""
    env_in = dict(os.environ if environ is None else environ)
    try:
        reg = R.with_kv_dtype(_load(name), env_in.get("KV_DTYPE"))
    except R.ProfileError as e:
        raise CliError(str(e)) from None
    # every instance gets the same KV host tier and PLE table (the checkpoint's: INT6 g32 with --ple int6)
    inv = replace(inv, kv_tier_gib=pf.kv_host_tier_gib(env_in), ple_table_gib=pf.ple_table_gib_of(path or env_in.get("MERLIN_CHECKPOINT_DIR"), inv.ple_table_gib, env_in.get("PLE_STORE")))
    if _gpu_selection(env_in) is not None:
        raise CliError("--instances partitions the host itself; unset GPUS / CUDA_VISIBLE_DEVICES (or start the instances by hand)")
    if n < 1:
        raise CliError("--instances needs a whole number >= 1")
    sets, class_notes = _allocate_instances(reg, name, inv, profile, tp, n)
    base_port = int(env_in.get("PORT") or _default_port(reg))
    if any(x.startswith(("--port", "--host")) for x in extra):
        raise CliError("--instances assigns each server its own port: set PORT=<base> instead of -- --port / --host")
    plans: list[Plan] = []
    used_ports: list[list[int]] = []
    for i, (p, cards) in enumerate(sets):
        e = dict(env_in)
        e["GPUS"] = e["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cards))
        port = base_port + i
        e["PORT"] = str(port)
        try:  # never a caller's export: the Super VL builder derives the same pair from PORT itself
            m, d = L.distributed_ports(str(port))
        except L.LaunchError as err:
            raise CliError(f"instance {i}: {err}; lower PORT", err.code) from None
        e["MASTER_PORT"], e["VLLM_PORT"] = str(m), str(d)
        cache_root = e.get("CACHE_ROOT") or f"{repo.as_posix()}/.vllm-cache"
        if i > 0:  # compile / Triton / topology caches of their own (the 27B builder exports no cache variables itself)
            e["CACHE_ROOT"] = e["VLLM_CACHE_ROOT"] = f"{cache_root}/instance-{i}"
            e["TRITON_CACHE_DIR"] = f"{e['CACHE_ROOT']}/triton"
        if served_suffix:
            if reg["model"] == "flash-next":
                ck = reg.get("checkpoints", {}).get(p.get("checkpoint") or "", {})  # Swift / Huihui serve their own names
                names = (e.get("MODEL_NAME") or ck.get("served_names") or reg["base"]["served_names"]).split()
                e["MODEL_NAME"] = " ".join(f"{x}-{i}" for x in names)
            else:
                default_served = reg["launch"]["served"] if reg["model"] in ("svl", "ornith") else reg["launch"]["checkpoints"][name]["served"]
                e["SERVED"] = f"{e.get('SERVED') or default_served}-{i}"
        plan = resolve(name, path, p["id"], inv, extra, repo, e, tp)
        plan.env.update({k: e[k] for k in ("PORT", "MASTER_PORT", "VLLM_PORT", "CACHE_ROOT", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR",
                                           "MODEL_NAME", "SERVED") if k in e})
        http = L.effective_host_port(plan.vllm.args, plan.vllm.extra)[1]  # type: ignore[union-attr]
        ports = [http, m, d, d + 1]
        if len(set(ports)) < len(ports):
            raise CliError(f"instance {i}: HTTP / MASTER_PORT / VLLM_PORT collide ({ports}); choose another PORT")
        for j, prev in enumerate(used_ports):
            shared = sorted(set(ports) & set(prev))
            if shared:
                raise CliError(f"instance {i} and instance {j} would share a port ({shared}); choose another PORT")
        used_ports.append(ports)
        plan.notes = [x for x in plan.notes if (not x.startswith("GPUs ") or "skipped" in x) and (i == 0 or not x.startswith("P2P is off"))]
        if i == 0 and any(x.startswith("P2P is off") for x in plan.notes):  # the throughput note counts the whole class
            first = next(g for g in inv.gpus if g.index == cards[0])
            n_class = len(_idle([g for g in inv.gpus if _class_of(g) == _class_of(first)]))
            plan.notes = [_throughput_note(reg, name, path or PLACEHOLDER, inv, p, cards, n_class)
                          if x.startswith("P2P is off") else x for x in plan.notes]
        plans.append(plan)
    used = {c for _, cards in sets for c in cards}
    port = base_port + n
    for group in _groups(inv):
        left = [g.index for g in _idle(group) if g.index not in used]
        mine = [(q, cards) for q, cards in sets if cards[0] in [g.index for g in group]]
        if left and mine:  # a class that got instances
            found = _hint_set(reg, name, inv, mine[0][0], group, left, tp, profile, beside=[len(c) for _, c in sets])
            if not isinstance(found, str):
                q, pick = found
                same = q is mine[0][0]
                plans[0].notes.append(f"GPUs {','.join(map(str, left))} free: {_instance_env_hint(repo, n, port)} "
                                      f"{_hint_cmd(name, path or PLACEHOLDER, pick, port, tp=tp if same else None, profile=profile if same else None)}")
                port += 1
            else:
                plans[0].notes.append(f"GPUs {','.join(map(str, left))} unused: no {name} layout for {len(left)} x {_class_name(group[0])} ({found})")
    plans[0].notes += class_notes
    if reg["model"] == "flash-next":
        need = max(p["match"]["cards"] for p, _ in sets)
        running = _running_fn_instances() // max(need, 1)
        tier = inv.kv_tier_gib
        floors = [pf.ram_floor_gib(len(cards), tier, inv.ple_table_gib) for _, cards in sets]  # an octo instance needs more than a quad
        ram, what, count = inv.host.ram_mib, "host RAM", n + running
        # the KV host tier's arenas (tier x ranks per server; running servers assumed to run the same layout and tier)
        tier_new = sum(tier * len(cards) for _, cards in sets)
        tier_gib = tier_new + running * tier * need
        need_gib = sum(floors) + running * (FN_RAM_FLOOR_GIB - pf.FN_PLE_TABLE_GIB + inv.ple_table_gib + tier * need)  # running: same tier and table
        avail = _mem_available_mib()
        if avail is not None and (ram is None or avail < ram) and inv.source == "probe":
            # what is free now, not MemTotal: running servers' PLE tables are already out of it, so only new ones count
            ram, what, count, need_gib, tier_gib = avail, "host memory available now", n, sum(floors), tier_new
        if ram is not None:
            have = pf._gib(ram)
            beside = f" beside {running} already running" if running else ""
            tier_s = f" + a KV host tier of {tier} GiB per rank ({tier_gib} GiB)" if tier > 0 else ""
            if have < need_gib:
                per = " / ".join(f"~{f:.0f}" for f in sorted(set(floors)))
                raise CliError(
                    f"{what} {have:.0f} GiB cannot hold {n} Flash-Next instance(s){beside}: each {pf.ple_pin_text(inv.ple_table_gib)} "
                    f"and needs {per} GiB resident ({FN_RAM_FLOOR_GIB:.0f} GiB at four workers plus "
                    f"{pf.RAM_PER_EXTRA_WORKER_GIB:.0f} GiB per worker above four{tier_s}; ~{need_gib:.0f} GiB for {count}; "
                    f"{FN_RAM_WARN_GIB:.0f} GiB per instance is the qualified allocation). Start fewer instances."
                )
            if not inv.ple_table_gib and have < need_gib + pf.PS.CACHE_WARN_GIB:  # the instances share one page-cache copy
                plans[0].warnings.append(
                    f"ram: {what} {have:.0f} GiB leaves under {pf.PS.CACHE_WARN_GIB} GiB of page cache for the PLE table above "
                    f"the computed ~{need_gib:.0f} GiB{tier_s}; decode and prefill slow down"
                )
            if inv.ple_table_gib and have < count * FN_RAM_WARN_GIB + tier_gib:
                plans[0].warnings.append(
                    f"ram: {what} {have:.0f} GiB is below {count} x {FN_RAM_WARN_GIB:.0f} GiB qualified{tier_s} "
                    f"(floor {need_gib:.0f} GiB; the boot peak is unmeasured)"
                )
    _check_kv_host_tier_instances(reg, inv, env_in, sets, n)
    return plans


def _kv_host_tier_state(env: dict[str, str]) -> tuple[str, int]:
    """MERLIN_KV_HOST_TIER_GIB as the launcher reads it (stripped, ASCII digits): ``("unset", 0)`` when absent or blank,
    ``("off", 0)`` for a genuine 0, ``("on", n)`` for n > 0 and ``("malformed", 0)`` for anything else (the launcher
    refuses that one; --swarm must never overwrite it)."""
    raw = (env.get("MERLIN_KV_HOST_TIER_GIB") or "").strip()
    if not raw:
        return "unset", 0
    if not (raw.isascii() and raw.isdigit() and len(raw) < 6):
        return "malformed", 0
    return ("on", int(raw)) if int(raw) > 0 else ("off", 0)


def _kv_host_tier_gib(env: dict[str, str]) -> int:
    """MERLIN_KV_HOST_TIER_GIB as the engine reads it; anything but a plain positive integer is 0 here (the engine
    itself refuses a malformed value)."""
    return _kv_host_tier_state(env)[1]


def _check_kv_host_tier_instances(reg: dict, inv: pf.Inventory, env: dict[str, str], sets: list, n: int) -> None:
    """The host-RAM KV tier knob is GiB per rank and every instance inherits it: each engine checks only its own ranks
    before weights load, so check the sum of every planned instance's arenas against the host before launching any
    (Flash-Next's own RAM gates already count its arenas with its resident floor)."""
    gib = _kv_host_tier_gib(env)
    if gib <= 0:
        return
    rank_counts = [len(cards) for _, cards in sets]
    minimum = max(pf.min_restore_unit_gib_per_rank(ranks, reg["model"], prof) for (prof, _), ranks in zip(sets, rank_counts))
    if gib < minimum:
        raise CliError(
            f"{KV_HOST_TIER_ENV}={gib} GiB per rank cannot hold one 262K restore unit "
            f"for this layout; needs at least {minimum} GiB per rank. "
            "Increase MERLIN_KV_HOST_TIER_GIB or set it to 0 to disable the tier."
        )
    if reg["model"] == "flash-next":
        # Its allocation and instance-count RAM gates already account for the arenas.
        return
    ranks = sum(rank_counts)
    tier_need = gib * ranks
    ram, what = inv.host.ram_mib, "host RAM"
    avail = _mem_available_mib()
    if avail is not None and (ram is None or avail < ram) and inv.source == "probe":
        ram, what = avail, "host memory available now"
    if ram is None:
        return
    have = pf._gib(ram)
    if have < tier_need:
        raise CliError(
            f"MERLIN_KV_HOST_TIER_GIB={gib} is GiB per rank: {n} instance(s) with {ranks} ranks in all need "
            f"{tier_need} GiB of host-RAM KV tier arenas, but {what} is {have:.0f} GiB. "
            "Lower MERLIN_KV_HOST_TIER_GIB, start fewer instances, or set it to 0 (tier off)."
        )


# --kv-ram-cache: the KV host tier (a host-RAM cache for evicted prefix KV blocks,
# restored instead of recomputed; no extra context or sessions) as the host RAM one server gives it in all, split
# evenly over that server's ranks (TP x PP) into the engine's knob, which is GiB per rank.
KV_HOST_TIER_ENV = "MERLIN_KV_HOST_TIER_GIB"
KV_TIER_FAMILY = "flash-next"  # the Flash-Next family (PLE table, DFlash2)
# The families whose RecoverSSM path the tier is qualified for (vllm/merlin/kv_host_tier.py): Flash-Next and the
# dense 27B lanes, where the cache is experimental, and Ornith 1.5 (Qwen3.5 MoE), whose hardware qualification has not
# run yet, so its plans say untested.
KV_TIER_FAMILIES = ("flash-next", "27b", "ornith")
# --swarm: the families that have rows with a qualified high-concurrency cap (profile ``serve.kv_tier_seqs``:
# Flash-Next 4 x 3090 24 sessions, 27B TP4 32). A row without that block refuses --swarm and names the rows that have it.
SWARM_FAMILIES = ("flash-next", "27b")
# The plan note per family besides Flash-Next (whose experimental note the launcher adds): 27B experimental, Ornith
# untested.
KV_TIER_UNTESTED = {"27b": "KV RAM cache on the 27B: experimental (RecoverSSM path only; a "
                           "session evicted under heavy concurrent prefill may recompute, output identical)",
                    "ornith": "KV RAM cache on Ornith 1.5: untested ⚫ (the engine accepts it on the RecoverSSM path; "
                              "its hardware qualification has not run yet)"}
# Above this --kv-ram-cache total (GiB per server) the plan adds an untested note. 48 GiB is the largest size with a
# recorded run; raise it only when a larger size has been run.
KV_RAM_CACHE_TESTED_GIB = 48
_KV_RAM_CACHE = re.compile(r"(\d+)(?:g|gib)?", re.IGNORECASE)


def parse_kv_ram_cache(text: str) -> int:
    """``--kv-ram-cache`` as whole GiB per server: ``24g``, ``24G``, ``24gib`` or a plain ``24``; 0 = off."""
    m = _KV_RAM_CACHE.fullmatch(str(text).strip())
    if not m:
        raise CliError(f"--kv-ram-cache {text!r}: give the KV RAM cache's host RAM per server in whole GiB, "
                       "e.g. 24g, 24G, 24gib or 24 (0 = off)")
    return int(m.group(1))


def kv_ram_cache_per_rank(total: int, ranks: list[int], minimum: int | None = None) -> int:
    """GiB per rank for one server's tier, rounded down to a whole GiB per rank. ``minimum`` is the GiB per rank one
    262K restore unit needs on these servers' rows (default: 4 GiB per server)."""
    if total <= 0:
        return 0
    sizes = sorted(set(ranks))
    if len(sizes) != 1:
        raise CliError(f"--kv-ram-cache {total}g: these servers have different rank counts ({' / '.join(map(str, sizes))}), so "
                       f"one size cannot split evenly over all of them; start one layout (--tp), or set {KV_HOST_TIER_ENV} "
                       "(GiB per rank) yourself")
    n = sizes[0]
    per_rank = total // n
    minimum = pf.min_restore_unit_gib_per_rank(n) if minimum is None else minimum
    if per_rank < minimum:
        smallest = minimum * n
        raise CliError(
            f"--kv-ram-cache {total}g cannot hold one 262K restore unit over the {n} ranks "
            f"(TP x PP) of each server; the smallest size is {smallest}g "
            f"({minimum} GiB per rank), or 0 (off). Nothing started."
        )
    return per_rank  # kv_ram_cache_text reports the size actually used


def kv_ram_cache_untested(total: int | None) -> str | None:
    """The NOTE for a --kv-ram-cache total above the tested size; None at or below it."""
    if not total or total <= KV_RAM_CACHE_TESTED_GIB:
        return None
    return f"--kv-ram-cache {total}g: sizes above {KV_RAM_CACHE_TESTED_GIB}g are untested ⚫ (no run has covered them)"


def kv_ram_cache_text(total: int, per_rank: int, ranks: int, servers: int = 1) -> str:
    """The plan line for the tier: per server, per rank and, with several servers, in all."""
    if total <= 0:
        return "KV RAM cache: off (--kv-ram-cache 0)"
    used = per_rank * ranks
    asked = f"{total}g rounded down to " if used != total else ""
    s = f"KV RAM cache: {asked}{used} GiB per server = {per_rank} GiB per rank x {ranks} ranks ({KV_HOST_TIER_ENV}={per_rank})"
    return s + (f"; {servers} servers = {used * servers} GiB of host RAM in all" if servers > 1 else "")


def _mem_available_mib() -> int | None:
    """MemAvailable capped by what the cgroup has left (``preflight.mem_available_mib``); None off Linux."""
    return pf.mem_available_mib()


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def cmd_list(out: Any, repo: Path = REPO, checkpoint: str | None = None, arch: str | None = None, cards: int | None = None,
             vram: int | None = None) -> int:
    """checkpoint -> profile -> status -> hardware -> launcher file, one row per profile; the filters narrow the rows to
    one checkpoint name, an arch (sm_80 / sm_86 / sm_89), a card count and a VRAM class."""
    if checkpoint and checkpoint not in CHECKPOINTS:
        raise CliError(f"unknown checkpoint {checkpoint!r}; known: {', '.join(CHECKPOINTS)}")
    out.write(f"{'checkpoint':<18} {'profile':<36} {'status':<9} {'hardware':<30} launcher file (shim for {LAUNCHER})\n")
    for name, key in CHECKPOINTS.items():
        if checkpoint and name != checkpoint:
            continue
        reg = R.load_registry(key)
        for p in _profiles_for(name, reg):
            m = p["match"]
            if (arch and arch not in m["arch"]) or (cards is not None and m["cards"] != cards) or (vram is not None and m["vram_gib"] != vram):
                continue
            hw = f"{m['cards']} x {m['vram_gib']} GB {'/'.join(m['arch'])} p2p={m['p2p']}"
            rep_launcher = SHIMS.get((name, p["id"]))
            if rep_launcher is None and key == "flash-next":
                rep_launcher = p.get("recipe_launcher") or (f"tools/merlin/launchers/{R.launcher_name(p['id'])}" if p["id"] in reg.get("launchers", []) else None)
            rep_launcher = rep_launcher or f"(merlin serve {name} --profile {p['id']})"
            where = rep_launcher if rep_launcher.startswith("(") or (repo / rep_launcher).is_file() else f"{rep_launcher}  [MISSING in this tree]"
            out.write(f"{name:<18} {p['id']:<36} {p['status']:<9} {hw:<30} {where}\n")
    return 0


def _print_plan(plan: Plan, out: Any, label: str) -> None:
    out.write(f"# {label}\n")
    if plan.vllm is not None:
        for note in plan.vllm.notes:
            if note in (R.SM80_FP8_WARNING, R.SM80_27B_FP8_WARNING):
                out.write(f"# WARN {note}\n")
            elif note.startswith("GUARD="):
                out.write(f"# NOTE {note}\n")
    for n in plan.notes:
        out.write(f"# NOTE {n}\n")
    for n in (plan.vllm.notes if plan.vllm is not None else []):
        if L.is_cap_note(n) or n.startswith("context:"):  # cap and context notes also print at boot
            out.write(f"# WARN {n.removeprefix('WARNING: ')}\n" if n.startswith("WARNING: ") else f"# NOTE {n.removeprefix('note: ')}\n")
    for w in plan.warnings:
        out.write(f"# WARN {w}\n")
    for m in missing_resources(plan, os.environ):
        out.write(f"# MISSING {m}\n")
    for k, v in plan.env.items():
        out.write(f"{k}={shlex.quote(v)} \\\n")
    out.write(" ".join(shlex.quote(a) for a in plan.argv) + "\n")
    if plan.vllm is not None:
        out.write("# resolves to:\n# " + " ".join(shlex.quote(a) for a in plan.vllm.argv()) + "\n")


def _identify(args: argparse.Namespace) -> tuple[str, str | None, list[str]]:
    targets = list(args.target)
    notes: list[str] = []
    if targets and targets[0] in CHECKPOINTS:
        name = targets[0]
        path = targets[1] if len(targets) > 1 else None
        if len(targets) > 2:
            raise CliError(f"unexpected argument {targets[2]!r} (put extra vllm args after --)")
    elif targets:
        path = targets[0]
        if len(targets) > 1:
            raise CliError(f"unexpected argument {targets[1]!r}: the first argument must be a checkpoint name ({', '.join(CHECKPOINTS)}) or a directory")
        if not Path(path).is_dir():
            raise CliError(f"{path!r} is neither a checkpoint name ({', '.join(CHECKPOINTS)}) nor a directory")
        try:
            name, notes = L.identify_checkpoint(path)
        except L.LaunchError as e:
            raise CliError(str(e), e.code) from None
        notes.insert(0, f"identified {path} as {name}")
    else:
        path = os.environ.get("MERLIN_CHECKPOINT_DIR")
        if not path:
            raise CliError("give the checkpoint directory (positional, or MERLIN_CHECKPOINT_DIR), or a checkpoint name")
        try:
            name, notes = L.identify_checkpoint(path)
        except L.LaunchError as e:
            raise CliError(str(e), e.code) from None
    return name, path, notes


# --spec: the drafter. mtp (the default everywhere) or dflash2, incoai's DFlash2 block-diffusion drafter for the
# Flash-Next family on the rows with a qualified DFlash2 geometry (+5-18 % decode vs MTP on 4 x 3090; a 789,668-token
# pool: 3 x 262K, not 6 x 131K). The launcher's SPEC=none (no drafter, a reference setting) is not offered here.
SPEC_CHOICES = ("mtp", "dflash2")
DFLASH2_REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
DFLASH2_DEFAULT_DIR = "~/models/flash-next-dflash2"  # setup.sh --spec dflash2 downloads it here
DFLASH2_FILES = ("config.json", "model.safetensors")  # what serving reads (the checksum list's files)
DFLASH2_WHY = "+5-18 % decode vs MTP on 4 x 3090; its 789,668-token pool holds 3 x 262K, not 6 x 131K"


def dflash2_dir(environ: dict[str, str] | None = None) -> Path:
    """The DFlash2 drafter's folder: DFLASH2_MODEL, else the default models folder's copy."""
    env = os.environ if environ is None else environ
    return Path(env.get("DFLASH2_MODEL") or DFLASH2_DEFAULT_DIR).expanduser()


def dflash2_missing(d: Path) -> list[str]:
    return [f for f in DFLASH2_FILES if not (d / f).is_file()]


def _spec_env(args: argparse.Namespace, name: str) -> dict[str, str]:
    """``--spec`` as the launcher's environment (SPEC, and DFLASH2_MODEL for dflash2); {} without the flag."""
    spec = getattr(args, "spec", None)
    if spec is None:
        return {}
    if CHECKPOINTS[name] != KV_TIER_FAMILY:
        if spec == "dflash2":
            raise CliError(f"--spec dflash2: DFlash2 is only available for the Flash-Next family ({L.FN_FAMILY_NAMES}); "
                           f"{name} uses its MTP head. Nothing started.")
        return {"SPEC": "mtp"}  # what it runs anyway
    if spec == "mtp":
        return {"SPEC": "mtp"}
    return {"SPEC": "dflash2", "DFLASH2_MODEL": str(dflash2_dir())}


def _serve_flags(args: argparse.Namespace, name: str, path: str | None) -> int | None:
    """``--ple int6`` and ``--kv-ram-cache``: refused before anything resolves when this build or this checkpoint
    cannot take them. Returns the tier's GiB per server (0 = off), or None without ``--kv-ram-cache``."""
    if getattr(args, "ple_store", None) == "disk" and CHECKPOINTS.get(name) != KV_TIER_FAMILY:
        raise CliError(f"--ple-store disk: {name} has no PLE table; it applies to the Flash-Next family only. Nothing started.")
    if getattr(args, "ple", "fp8") == "int6":
        import setup_cli

        why = setup_cli.int6_reason(name, REPO)
        if why is None and path is not None:
            why = setup_cli.int6_checkpoint_reason(Path(path)) or setup_cli.overlay_reason(name, Path(path), REPO)
        if why is not None:
            raise CliError(f"--ple int6: {why}. Nothing started; --ple fp8 (the default) serves the FP8 PLE table.")
    elif path is not None and CHECKPOINTS.get(name) == KV_TIER_FAMILY:
        import setup_cli

        if setup_cli.int6_checkpoint_reason(Path(path)) is None:  # the launcher boots INT6 g32 from the config alone
            raise CliError(f"{path} carries the INT6 g32 PLE table: serve it with --ple int6. Nothing started; "
                           "--ple fp8 (the default) needs the FP8 checkpoint.")
    if getattr(args, "kv_ram_cache", None) is None:
        return None
    total = parse_kv_ram_cache(args.kv_ram_cache)
    if total and CHECKPOINTS[name] not in KV_TIER_FAMILIES:
        raise CliError("--kv-ram-cache: the KV RAM cache is available for the Flash-Next family, the 27B and Ornith 1.5 "
                       f"only, not {name}. Nothing started.")
    return total


def swarm_cache_set_text(total: int, need: int, ranks: int, cap: int) -> str:
    """The note for --swarm with no --kv-ram-cache: the cache was sized for the cap."""
    return (f"--swarm: no --kv-ram-cache was given, so the KV RAM cache is {total}g per server ({need} GiB per rank x {ranks} "
            f"ranks), the size the {cap}-session cap was tested at; --kv-ram-cache SIZE sets another size (at least {total}g)")


def _restore_unit_minimum(plans: list[Plan]) -> int:
    """GiB per rank one 262K restore unit needs on every planned server (each row's own host row size)."""
    return max(pf.min_restore_unit_gib_per_rank(len(p.cards), p.vllm.model if p.vllm else None,
                                                p.vllm.profile if p.vllm else None) for p in plans)


def _plan_kv_ram_cache(plan: Callable[[dict[str, str] | None], list[Plan]], total: int | None,
                       base: dict[str, str] | None = None, swarm: bool = False) -> list[Plan]:
    """Resolve with the tier off to learn each server's ranks, split ``total`` over them (rounded down to whole GiB per
    rank; refused below one 262K restore unit), then resolve again with the per-rank knob so every RAM gate counts the tier; the layout must not move.

    With ``swarm`` the layout is learned at the default cap, then the cache is checked against the size the
    row's swarm cap was qualified with: ``total`` below it is refused (never raised), and ``total`` None (no
    --kv-ram-cache given) is set to it, per server, with a note."""
    env = {**os.environ, **(base or {}), KV_HOST_TIER_ENV: "0"}  # the flag wins over an exported knob
    env.pop(L.SWARM_ENV, None)  # the layout (and its ranks) is learned at the default cap
    # knobs that need the tier on are left out of the tier-off layout pass and checked in the sized pass below
    needs_tier = {k: env.pop(k) for k in L.FN_KV_HOST_TIER_NEEDS_ON if k in env}
    plans = plan(env)
    ranks = [len(p.cards) for p in plans]
    swarm_note = None
    if swarm:
        prof = plans[0].vllm.profile
        gate = L.swarm_gate(prof)
        if not gate:
            raise CliError(L.swarm_not_qualified(plans[0].profile_id) + ". Nothing started.")
        sizes = sorted(set(ranks))
        if len(sizes) != 1:
            raise CliError(f"--swarm: these servers have different rank counts ({' / '.join(map(str, sizes))}), so one KV RAM "
                           "cache size cannot suit all of them; start one layout (--tp). Nothing started.")
        need, cap = int(gate["min_gib_per_rank"]), int(gate["max_num_seqs"])
        if total is None:
            total = need * sizes[0]
            swarm_note = swarm_cache_set_text(total, need, sizes[0], cap)
        elif total // sizes[0] < need:
            raise CliError(L.swarm_cache_refusal(prof, total // sizes[0]) + ". Nothing started.")
        env[L.SWARM_ENV] = "1"
    assert total is not None
    per = kv_ram_cache_per_rank(total, ranks, _restore_unit_minimum(plans))
    env.update(needs_tier)
    if per or needs_tier:  # with --kv-ram-cache 0 those knobs are refused as the launcher refuses them
        env[KV_HOST_TIER_ENV] = str(per)
        again = plan(env)
        if [len(p.cards) for p in again] != ranks:
            raise CliError(f"--kv-ram-cache {total}g changes the layout this host gets; give --tp to pick one. Nothing started.")
        plans = again
    for p in plans:
        p.env[KV_HOST_TIER_ENV] = str(per)
    plans[0].notes.append(kv_ram_cache_text(total, per, ranks[0], len(plans)))
    if swarm_note:
        plans[0].notes.append(swarm_note)
    big = kv_ram_cache_untested(total)
    if big:
        plans[0].notes.append(big)
    return plans


def _swarm_env(args: argparse.Namespace, name: str) -> dict[str, str]:
    """The one swarm decision as the launcher's environment: ``--swarm`` or an exported SWARM=1 gives SWARM=1 (the plan, the
    cache sizing and the hand-over to the launcher all read this and nothing else); otherwise {}. A bad exported value is
    refused here. Refused where the checkpoint's family has no swarm cap at all, naming the rows that have one (a row of a
    swarm family without one is refused by the launcher, which knows the row)."""
    try:
        wanted = bool(getattr(args, "swarm", False)) or L.swarm_requested(os.environ)
    except L.LaunchError as e:
        raise CliError(str(e), e.code) from None
    if not wanted:
        return {}
    if CHECKPOINTS[name] not in SWARM_FAMILIES:
        raise CliError(L.swarm_not_qualified(name) + ". Nothing started.")
    return {L.SWARM_ENV: "1"}


def _verification_list(name: str) -> tuple[Path, dict[str, str]]:
    import setup_cli

    lst = setup_cli.checksum_list(name, REPO)
    sums, _ = setup_cli.parse_checksums(lst)
    if not any(n.endswith(".safetensors") for n in sums):
        raise CliError(f"{lst}: no safetensors shards listed; cannot verify this checkpoint")
    return lst, sums


def verification_hint(name: str, path: str, ple: str = "fp8") -> str | None:
    """Check only the checksum list and file stats; never read shard bytes during serve."""
    import setup_cli

    errors = (CliError, setup_cli.SetupError, OSError, ValueError, TypeError, AttributeError)
    if ple != "fp8":
        try:  # the INT6 overlay: its base and pack carry the markers (serve refuses a stale overlay before this)
            why = setup_cli.overlay_reason(name, Path(path), REPO)
        except errors as e:
            why = str(e)
        return None if why is None else f"INT6 overlay unverified: {why}; run merlin verify {shlex.quote(path)} --checkpoint {name} --ple int6"
    try:
        lst, sums = _verification_list(name)
    except errors as e:
        # merlin verify reads this same list, so advising it would only fail (a withheld checkpoint ships none)
        reason = setup_cli.SETUP_WITHHELD.get(name) or str(e)
        return f"checkpoint shards are unverified; verification is unavailable for {name}: {reason}"
    try:
        if setup_cli.marker_ok(Path(path), name, sums, setup_cli.sha256_file(lst)):
            return None
    except errors:
        pass  # unreadable/malformed markers are unverified, not a launch refusal
    cmd = ["merlin", "verify", path, "--checkpoint", name]
    return f"checkpoint shards are unverified (no current size/mtime stamp); run {shlex.join(cmd)}"


def _verify_overlay(name: str, path: Path, out: Any) -> int:
    """``merlin verify <overlay> --ple int6``: rehash the FP8 base and the pack its marker names, then check the overlay
    against both (links, generated config.json / index, the checksum lists and base files it was built from)."""
    import ple_pack
    import setup_cli

    m = ple_pack.read_overlay_marker(path)
    if m is None:
        raise CliError(f"{path} is not an INT6 overlay (no {ple_pack.OVERLAY_MARKER}); --ple int6 verifies the folder setup built "
                       "(<checkpoint>-int6). Nothing verified")
    if m.get("model") != name:
        raise CliError(f"{path} is the INT6 overlay of {m.get('model')}, not {name}")
    base, pack = Path(str(m.get("base"))), Path(str(m.get("pack")))
    say = lambda line: print(line, file=out, flush=True)  # noqa: E731
    lst, sums = _verification_list(name)
    ok, bad = setup_cli.verify_checkpoint(base, lst, say=say, model=name, rehash=True)
    if not ok:
        raise CliError(f"FP8 base verification failed in {base}:\n  " + "\n  ".join(bad), 1)
    plst = setup_cli.pack_list(REPO)
    if not plst.is_file():
        raise CliError(f"{ple_pack.PACK_LIST} is missing from this release: the INT6 pack cannot be verified", 1)
    ok, bad = setup_cli.verify_checkpoint(pack, plst, say=say, model=ple_pack.PACK_MARKER_MODEL, rehash=True)
    if not ok:
        raise CliError(f"INT6 pack verification failed in {pack}:\n  " + "\n  ".join(bad), 1)
    probs = ple_pack.overlay_problems(path, setup_cli.overlay_sources(name, base, pack, REPO))
    if probs:
        raise CliError(f"INT6 overlay {path} is stale:\n  " + "\n  ".join(probs)
                       + f"\n  merlin pull {name} --models-dir {shlex.quote(str(base))} --ple int6 rebuilds it", 1)
    print(f"merlin: INT6 overlay verified: {len(sums)} FP8 files in {base}, the pack in {pack}, links and generated files in {path}", file=out)
    return 0


def cmd_verify(args: argparse.Namespace, out: Any) -> int:
    """Always rehash, including when a pull/verify stamp is current, without probing GPUs or installing anything."""
    import setup_cli

    try:
        path = Path(args.directory)
        if not path.is_dir():
            raise CliError(f"checkpoint directory {str(path)!r} is not a directory")
        name = args.checkpoint
        if name is None:
            name, notes = L.identify_checkpoint(str(path), REPO)
            for note in notes:
                print(f"merlin: NOTE {note}", file=out)
        if args.ple == "int6":
            return _verify_overlay(name, path, out)
        lst, sums = _verification_list(name)
        ok, bad = setup_cli.verify_checkpoint(
            path, lst, say=lambda line: print(line, file=out, flush=True), model=name, rehash=True,
        )
        if not ok:
            raise CliError(f"checkpoint verification failed in {path}:\n  " + "\n  ".join(bad), 1)
        print(f"merlin: checkpoint verified: {len(sums)} files in {path}", file=out)
        return 0
    except (CliError, L.LaunchError, setup_cli.SetupError) as e:
        print(f"merlin: {e}", file=sys.stderr)
        return e.code
    except (OSError, ValueError) as e:
        print(f"merlin: {e}", file=sys.stderr)
        return 2


def cmd_serve(args: argparse.Namespace, extra: list[str], out: Any) -> int:
    if args.simulate and not args.dry_run:
        print("merlin: --simulate only works with --dry-run (it never launches on fabricated hardware)", file=sys.stderr)
        return 2
    if args.profile and args.tp is not None:
        print("merlin: give --profile or --tp, not both", file=sys.stderr)
        return 2
    try:
        extra = L.C.context_args(getattr(args, "context", None), extra, L._flag_values(extra))
        name, path, id_notes = _identify(args)
        if path is not None and not Path(path).is_dir():
            raise CliError(f"checkpoint directory {path!r} is not a directory")
        if path is None and not os.environ.get("MERLIN_CHECKPOINT_DIR"):
            id_notes.append("no checkpoint directory given: the command below shows " + PLACEHOLDER + " (dry-run only)")
        tier = _serve_flags(args, name, path or os.environ.get("MERLIN_CHECKPOINT_DIR"))  # the folder that will be served
        swarm_env = _swarm_env(args, name)
        ckpt = path or os.environ.get("MERLIN_CHECKPOINT_DIR")
        hint = verification_hint(name, ckpt, args.ple) if ckpt else None
        if hint:
            id_notes.append(hint)
        spec_env = _spec_env(args, name)
        launch_env = {**spec_env, **swarm_env}  # what the launcher is told (SPEC, SWARM); the plan env gets it at exec
        if args.ple_store:
            launch_env["PLE_STORE"] = args.ple_store
        if not swarm_env and L.SWARM_ENV in os.environ:
            launch_env[L.SWARM_ENV] = "0"  # decided off: the plan and the launcher read the same 0 whatever the export spelt
        inv = pf.simulate(args.simulate) if args.simulate else pf.probe(pf.default_runner)

        def plan(environ: dict[str, str] | None) -> list[Plan]:
            if args.instances is not None and args.instances != 1:
                return plan_instances(name, path, args.profile, inv, extra, args.instances, repo=REPO, environ=environ,
                                      tp=args.tp, served_suffix=args.served_suffix)
            return [resolve(name, path, args.profile, inv, extra, repo=REPO, environ=environ, tp=args.tp)]

        if tier is not None or (swarm_env and _kv_host_tier_state(os.environ)[0] in ("unset", "off")):
            # --kv-ram-cache SIZE, or --swarm with no cache anywhere (unset, blank or a genuine 0): sized for the swarm cap
            # An exported size, a small or a malformed one included, goes to the launcher as it is: never overwritten
            plans = _plan_kv_ram_cache(plan, tier, launch_env, swarm=bool(swarm_env))
        else:
            plans = plan({**os.environ, **launch_env} if launch_env else None)
            inherited_tier = _kv_host_tier_gib({**os.environ, **launch_env})
            if inherited_tier > 0:
                minimum = _restore_unit_minimum(plans)
                if inherited_tier < minimum:
                    raise CliError(
                        f"{KV_HOST_TIER_ENV}={inherited_tier} GiB per rank cannot hold one "
                        f"262K restore unit for this layout; needs at least {minimum} GiB per rank. "
                        f"Increase {KV_HOST_TIER_ENV} or set it to 0 to disable the tier."
                    )
        # The untested notice follows the tier the launcher actually gets (the plan's env over the inherited one):
        # --kv-ram-cache or an exported MERLIN_KV_HOST_TIER_GIB alike
        if CHECKPOINTS[name] in KV_TIER_UNTESTED and any(
                _kv_host_tier_gib({**os.environ, **p.env}) > 0 for p in plans):
            plans[0].notes.append(KV_TIER_UNTESTED[CHECKPOINTS[name]])
        for p in plans:
            p.env.update(launch_env)
        if spec_env.get("SPEC") == "dflash2":
            d = Path(spec_env["DFLASH2_MODEL"])
            gone = dflash2_missing(d)
            fix = (f"the DFlash2 drafter is not in {d} (missing {', '.join(gone)}): ./setup.sh --spec dflash2 downloads "
                   f"{DFLASH2_REPO} there, or set DFLASH2_MODEL to your copy")
            if gone and not args.dry_run:
                raise CliError(fix + ". Nothing started.")
            plans[0].notes.append(f"--spec dflash2: drafter {d}; {DFLASH2_WHY}")
            if gone:
                plans[0].warnings.append("dflash2: " + fix)
    except CliError as e:
        print(f"merlin: {e}", file=sys.stderr)
        return e.code
    except (ValueError, OSError, KeyError) as e:
        print(f"merlin: {e}", file=sys.stderr)
        return 2
    plans[0].notes[:0] = id_notes
    if args.dry_run:
        for i, plan in enumerate(plans):
            cards = ",".join(map(str, plan.cards))
            label = f"{plan.checkpoint} -> {plan.profile_id} [{plan.status}] on GPUs {cards} -> {plan.launcher}"
            if len(plans) > 1:
                label = f"instance {i}: " + label + f" port {plan.env.get('PORT')}"
            _print_plan(plan, out, label)
        return 0
    for plan in plans:
        if PLACEHOLDER in plan.argv:
            print("merlin: give the checkpoint directory (positional, or MERLIN_CHECKPOINT_DIR)", file=sys.stderr)
            return 2
        gone = missing_resources(plan, os.environ)
        if gone:
            print("merlin: missing launch resources, nothing started:\n  " + "\n  ".join(gone), file=sys.stderr)
            return 2
    sys.stdout.flush()
    if len(plans) == 1:
        plan = plans[0]
        print(f"merlin: {plan.checkpoint} -> {plan.profile_id} [{plan.status}] on GPUs {','.join(map(str, plan.cards))} -> {plan.launcher}", file=sys.stderr)
        for n in plan.notes:
            print(f"merlin: NOTE {n}", file=sys.stderr)
        for w in plan.warnings:
            print(f"merlin: WARN {w}", file=sys.stderr)
        os.execvpe(plan.argv[0], plan.argv, {**os.environ, **plan.env})
        return 0  # pragma: no cover
    return launch_instances(plans, Path(os.environ.get("MERLIN_LOG_DIR") or os.getcwd()), wait_ready=getattr(args, "wait_ready", 0.0) or 0.0)


def _plan_ports(plan: Plan) -> list[tuple[str, int]]:
    """(label, port) of every port one instance binds: its HTTP port and the MASTER_PORT / VLLM_PORT pair."""
    out = [("HTTP port", L.effective_host_port(plan.vllm.args, plan.vllm.extra)[1])]  # type: ignore[union-attr]
    for key in ("MASTER_PORT", "VLLM_PORT"):
        v = plan.env.get(key)
        if v and v.isdigit():
            out.append((key, int(v)))
    return out


def _stop_instances(procs: list[subprocess.Popen], grace: float = 10.0) -> list[int]:
    """End the process groups this invocation spawned (SIGTERM, then SIGKILL after ``grace`` s) and reap them.
    A group counts as gone only when no process is left in it: a worker that ignores SIGTERM after its leader
    exited still gets SIGKILL. Returns the group ids that could not be confirmed gone. Never touches a process
    it was not handed."""
    import signal
    import time

    def send(proc: subprocess.Popen, sig: int) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, sig)  # start_new_session=True made the child its own group leader
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def alive(proc: subprocess.Popen) -> bool:
        """The leader is reaped first (a zombie leader would keep its group alive); then signal 0 to the group
        says whether any process is left in it."""
        leader = proc.poll() is None
        if not hasattr(os, "killpg"):
            return leader
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return False
        except OSError:  # PermissionError: something is in the group that this user cannot signal
            return True
        return True

    def wait_gone(group: list[subprocess.Popen], until: float) -> list[subprocess.Popen]:
        left = [p for p in group if alive(p)]
        while left and time.monotonic() < until:
            time.sleep(0.2)
            left = [p for p in left if alive(p)]
        return left

    for proc in procs:
        send(proc, signal.SIGTERM)
    left = wait_gone(list(procs), time.monotonic() + grace)
    for proc in left:
        send(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    for proc in left:
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - report, never raise out of a rollback
            pass
    return [p.pid for p in wait_gone(left, time.monotonic() + 5)]


def _health_ok(host: str, port: int) -> bool:
    import urllib.request

    h = host.strip("[]")
    h = "127.0.0.1" if h in ("", "0.0.0.0") else h
    h = f"[{h}]" if ":" in h else h
    try:
        # never through an http_proxy from the environment: a missed answer here stops every instance at the deadline
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://{h}:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 - not up yet
        return False


def launch_instances(plans: list[Plan], log_dir: Path, wait_ready: float = 0.0) -> int:
    """Own the instances in the foreground, from spawn through shutdown.
    All ports are checked before spawning. Startup/readiness failure, an instance
    exit, Ctrl+C or SIGTERM stops and reaps this invocation's process groups.
    """
    import signal
    import time

    hosts = [L.effective_host_port(p.vllm.args, p.vllm.extra)[0] for p in plans]  # type: ignore[union-attr]
    try:
        for i, (plan, host) in enumerate(zip(plans, hosts)):
            for label, port in _plan_ports(plan):
                if not L.port_free(host, port):
                    hint = f" (PORT={port + len(plans)} moves the whole block)" if label == "HTTP port" else ""
                    print(f"merlin: instance {i}: {label} {port} on {host} is busy; nothing started{hint}", file=sys.stderr)
                    return 2
    except L.LaunchError as e:
        print(f"merlin: {e}; nothing started", file=sys.stderr)
        return e.code
    spawned: list[tuple[int, Plan, int, subprocess.Popen, Path]] = []
    interrupted = 0
    previous_handlers = {}

    def request_stop(signum, _frame):
        # Defer interruption until Popen has returned and its child is recorded.
        # Repeated signals during cleanup must not interrupt group termination.
        nonlocal interrupted
        interrupted = interrupted or signum

    def check_interrupted():
        if interrupted:
            raise KeyboardInterrupt

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):  # SIGHUP: an ssh drop; each instance has its own session and never gets it
            previous_handlers[sig] = signal.signal(sig, request_stop)
        try:
            for i, plan in enumerate(plans):
                check_interrupted()
                env = {**os.environ, **plan.env}
                port = _plan_ports(plan)[0][1]
                log = log_dir / f"merlin-{plan.profile_id}-{port}.log"
                with open(log, "ab") as fh:
                    proc = subprocess.Popen(plan.argv, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
                    spawned.append((i, plan, port, proc, log))
                check_interrupted()
                print(f"merlin: instance {i}: {plan.profile_id} [{plan.status}] GPUs {','.join(map(str, plan.cards))} -> http://{hosts[i]}:{port} pid {proc.pid} (own process group) log {log}", file=sys.stderr)
                for n in plan.notes:
                    print(f"merlin: NOTE {n}", file=sys.stderr)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            print(f"merlin: instance {len(spawned)} failed to start ({e}); stopping this command's instances",
                  file=sys.stderr)
            return 1
        if wait_ready > 0:
            deadline = time.monotonic() + wait_ready
            pending = list(spawned)
            answered: set[int] = set()
            failure = ""
            while not failure:
                check_interrupted()
                for i, _plan, port, proc, log in spawned:  # every instance, also the ones that already answered
                    if proc.poll() is not None:
                        when = ("after it answered /health, while others were still loading" if i in answered
                                else "before it was ready")
                        failure = f"instance {i} (pid {proc.pid}) exited with code {proc.returncode} {when}; see {log}"
                        break
                if failure:
                    break
                for item in list(pending):
                    check_interrupted()
                    if _health_ok(hosts[item[0]], item[2]):
                        pending.remove(item)
                        answered.add(item[0])
                if not pending:
                    # ready only when every instance answers in one pass and every process is still running after it
                    pending = [s for s in spawned if not _health_ok(hosts[s[0]], s[2])]
                    if any(s[3].poll() is not None for s in spawned):
                        continue  # the exit is reported at the top of the loop
                    if not pending:
                        break
                if time.monotonic() >= deadline:
                    failure = f"instance(s) {', '.join(str(x[0]) for x in pending)} not ready after {wait_ready:g}s"
                else:
                    time.sleep(1.0)
            if failure:
                print(f"merlin: {failure}; stopping this command's instances", file=sys.stderr)
                return 1
            check_interrupted()
            print("merlin: ready: " + " ".join(f"http://{hosts[i]}:{port}" for i, _, port, _, _ in spawned), file=sys.stderr)
        else:
            print("merlin: spawned (not yet ready; poll /health, load takes minutes): " + " ".join(f"http://{hosts[i]}:{port}" for i, _, port, _, _ in spawned), file=sys.stderr)
        print("merlin: running in foreground; Ctrl+C stops all instances", file=sys.stderr)
        while spawned:
            check_interrupted()
            for i, _plan, port, proc, log in spawned:
                if proc.poll() is not None:
                    print(f"merlin: instance {i} (pid {proc.pid}) exited with code {proc.returncode}; "
                          f"stopping this command's instances; see {log}", file=sys.stderr)
                    return 1
            time.sleep(0.2)
        return 0
    except KeyboardInterrupt:
        signum = interrupted or signal.SIGINT
        print(f"merlin: received {signal.Signals(signum).name}; stopping this command's instances", file=sys.stderr)
        return 128 + signum
    finally:
        try:
            survivors = _stop_instances([s[3] for s in spawned])
            if survivors:
                print(f"merlin: could NOT confirm process group(s) {survivors} gone, stop them by hand",
                      file=sys.stderr)
            else:
                print(f"merlin: stopped the {len(spawned)} instance(s) this command had spawned. "
                      "Nothing is left running by this command.", file=sys.stderr)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


# merlin doctor --report: one redacted file for bug reports. Secrets never leave the host: the API key, Hugging
# Face tokens and auth headers are replaced before anything is written.
REPORT_LINES = 200  # server-log lines in the report
# A name that ends in a secret word (so "max_num_batched_tokens=8192" survives), then = or : (the name may be quoted, as in
# JSON and Python-dict logs), then the whole value: shell words, quoted strings (shlex.quote splits a value that holds a
# quote into several) and bracketed lists.
_NAME = (r"[A-Za-z0-9_.-]*(?:API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|SECRET[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?"
         r"|SIGNATURE|WATERMARK[_-]?CONFIG)")
_QUOTED = r"""(?:'[^']*'|"(?:[^"\\]|\\.)*"|\[[^\]]*\])"""
_WORD = rf"""(?:{_QUOTED}+|[^\s'"\[]\S*)"""
_SECRETS = [
    (re.compile(rf"""(?i)(authorization["']?\s*[=:]\s*)(?:(?:bearer|basic)\s+)?{_WORD}"""), r"\1<redacted>"),
    (re.compile(rf"""(?i)\b({_NAME})(["']?\s*[=:]\s*){_WORD}"""), r"\1\2<redacted>"),
    # --api-key takes several values on one line, in shell and in a printed argv list (quote-comma separators)
    (re.compile(rf"""(?i)(--api[-_]key(?:=|["']?,?\s+)){_WORD}(?:,?[ \t]+(?!["']?-|\S*://){_WORD})*"""), r"\1<redacted>"),
    (re.compile(rf"(?i)(--(?:hf[-_]token|watermark[-_]config|[a-z_-]*(?:token|secret|password|passwd))(?:=|\s+)){_WORD}"), r"\1<redacted>"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}"), r"\1 <redacted>"),
    (re.compile(r"(://[^/\s:@]+:)[^@\s/]+@"), r"\1<redacted>@"),  # user:password@host in a URL
    (re.compile(r"\b(?:hf|sk|ghp|gho|ghs|github_pat)[_-][A-Za-z0-9_-]{16,}\b"), "<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
]


# Values of settings named like secrets (HF_HUB_DISABLE_TOKEN=1) that are not secrets, and would blank out the report.
_NOT_SECRETS = frozenset(("", "0", "1", "true", "false", "yes", "no", "on", "off", "none", "null", "auto"))


def redact(text: str, literals: list[str] | tuple[str, ...] = ()) -> str:
    """``text`` with every known secret replaced: the literal values given (the live key and tokens) and anything shaped
    like a key, token or auth header."""
    for v in sorted({v.strip() for v in literals if v and v.strip().lower() not in _NOT_SECRETS}, key=len, reverse=True):
        # a short value only as a whole word: "abc" must not eat "abcdef", but a key that short still goes
        text = text.replace(v, "<redacted>") if len(v) >= 6 else re.sub(rf"(?<![\w-]){re.escape(v)}(?![\w-])", "<redacted>", text)
    for pat, sub in _SECRETS:
        text = pat.sub(sub, text)
    return text


def _secret_literals() -> list[str]:
    """The live secrets: env values whose names say key, token, secret, password or credential, the saved API key and the
    Hugging Face token file."""
    vals = [v for k, v in os.environ.items() if re.fullmatch(_NAME, k, re.I)]  # the same names the patterns redact
    hf = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    for f in (REPO / ".merlin" / "api-key", hf / "token"):
        try:
            vals.append(f.read_text())
        except OSError:
            pass
    return vals


def _newest_log() -> Path | None:
    d = Path(os.environ.get("MERLIN_LOG_DIR") or os.getcwd())
    logs = sorted(d.glob("merlin-*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def _tail(path: Path, n: int) -> str:
    from collections import deque

    with open(path, encoding="utf-8", errors="replace") as fh:
        return "".join(deque(fh, maxlen=n))


def doctor_report(rest: list[str], out: Any) -> int:
    """``merlin doctor --report [FILE] [--log FILE] [--lines N] [preflight args]``: preflight (host, GPUs, driver,
    topology, RAM), the resolved serve plan (profile and command, a dry run), and the server log's last lines, redacted,
    in one file (mode 0600)."""
    import contextlib
    import io
    import platform
    import time

    path, log, lines, fwd = None, None, REPORT_LINES, []
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--report":
            if i + 1 < len(rest) and not rest[i + 1].startswith("-"):
                path, i = Path(rest[i + 1]), i + 1
        elif a == "--log" and i + 1 < len(rest):
            log, i = Path(rest[i + 1]), i + 1
        elif a == "--lines" and i + 1 < len(rest) and rest[i + 1].isdigit():
            lines, i = int(rest[i + 1]), i + 1
        else:
            fwd.append(a)
        i += 1
    model = next((fwd[j + 1] for j, a in enumerate(fwd[:-1]) if a == "--model"), "flash-next")
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = path or Path(os.getcwd()) / f"merlin-report-{stamp}.txt"
    parts = [f"# merlin doctor --report ({stamp} UTC)",
             f"python {platform.python_version()} \u00b7 {platform.platform()} \u00b7 engine tree {REPO}"]
    for name in sorted(CHECKPOINTS):
        v = REPO / L.CHECKPOINTS[name][1] / "VERSION" if len(L.CHECKPOINTS[name]) > 1 else None
        if v is not None and v.is_file():
            parts.append(f"{name} recipe version {v.read_text().strip()}")
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            rc = pf.main(fwd, stdout=buf)
        except SystemExit as e:  # argparse, on an argument preflight does not know: the report still gets written
            rc = e.code if isinstance(e.code, int) else 2
    parts += ["", f"## preflight (exit {rc})", buf.getvalue().rstrip(), err.getvalue().rstrip()]
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            sim = next((fwd[j + 1] for j, a in enumerate(fwd[:-1]) if a == "--simulate"), None)
            rc = main(["serve", pf.CHECKPOINT_NAMES.get(model, model), "--dry-run", *(["--simulate", sim] if sim else [])], buf)
        except SystemExit as e:  # argparse
            rc = e.code if isinstance(e.code, int) else 2
    parts += ["", f"## serve plan, dry run (exit {rc})", err.getvalue().rstrip(), buf.getvalue().rstrip()]
    log = log or _newest_log()
    if log is not None and log.is_file():
        parts += ["", f"## server log: last {lines} lines of {log}", _tail(log, lines).rstrip()]
    else:
        parts += ["", "## server log", "none found (merlin-*.log in MERLIN_LOG_DIR or here); give --log FILE"]
    text = redact("\n".join(p for p in parts if p is not None) + "\n", _secret_literals())
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)  # mode applies only on create: force it before any secret lands
    if hasattr(os, "fchmod"):
        os.fchmod(fd, 0o600)
    os.ftruncate(fd, 0)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"merlin: report written to {path} (secrets redacted; read it before you share it)", file=out)
    return 0


def main(argv: list[str] | None = None, out: Any = None) -> int:
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["doctor"] and "--report" in argv[1:]:
        return doctor_report(argv[1:], out)
    if argv[:1] == ["doctor"]:  # transparent forwarding to preflight (options included)
        rest = argv[1:]
        return pf.main(rest[1:] if rest[:1] == ["--"] else rest, stdout=out)
    if argv[:1] == ["pull"]:  # download + verify a checkpoint (setup_cli does it; the engine is installed first when missing)
        import setup_cli

        if len(argv) < 2 or argv[1] not in CHECKPOINTS:
            print(f"merlin: pull needs a checkpoint name ({', '.join(CHECKPOINTS)}) [--models-dir DIR]", file=sys.stderr)
            return 2
        return setup_cli.main(["--pull-only", "--yes", "--model", argv[1], *argv[2:]], out=out)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="merlin", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve a checkpoint through the one launcher")
    s.add_argument("target", nargs="*", help="[checkpoint name] [checkpoint directory]; a directory alone is identified")
    s.add_argument("--profile", help="force a profile id of this checkpoint (checked against the hardware)")
    s.add_argument("--tp", type=int, help="pick the profile with this tensor-parallel size (a subset of a larger host)")
    s.add_argument("--instances", type=int, help="run N servers in the foreground on N disjoint card sets "
                   "(consecutive ports from PORT); Ctrl+C stops all instances")
    s.add_argument("--wait-ready", type=float, default=0.0, metavar="SECONDS",
                   help="with --instances: wait up to SECONDS for every /health; on a dead or late instance stop all of them and exit 1")
    s.add_argument("--served-suffix", action="store_true", help="with --instances: append -<i> to the served model name(s)")
    s.add_argument("--ple", choices=("fp8", "int6"), default="fp8", help="Flash-Next: the PLE table, fp8 (default) or int6 (INT6 "
                   "g32, 38.74 GiB: give the INT6 overlay folder that setup or merlin pull --ple int6 built, <checkpoint>-int6)")
    s.add_argument("--ple-store", choices=("ram", "disk"), default=None, help="Flash-Next: where the PLE table lives, ram (the default) or "
                   "disk (experimental: read from the checkpoint's own files, on NVMe; frees the table's host RAM, but first reads are much "
                   "slower; see docs/how-to-serve.md). Other storage warns and runs")
    s.add_argument("--spec", choices=SPEC_CHOICES, help="the drafter: mtp (default) or dflash2 (Flash-Next family, rows "
                   f"with a DFlash2 geometry; {DFLASH2_WHY.replace(chr(37), chr(37) * 2)}; the drafter from DFLASH2_MODEL or {DFLASH2_DEFAULT_DIR})")
    s.add_argument("--kv-ram-cache", metavar="SIZE", help="Flash-Next (experimental, MTP only), 27B (experimental) and Ornith 1.5: a host-RAM cache for prefix KV blocks the GPU pool evicts "
                   "(restored instead of recomputed; adds no context and does not change the session cap (--swarm does); RAM only, never disk; ~66K tokens/GiB on Flash-Next TP2xPP2; needs memlock: LimitMEMLOCK=infinity or ulimit -l unlimited). SIZE = host RAM per "
                   f"server (24g, 24gib or 24 = GiB; 0 = off), split over its TP x PP ranks into {KV_HOST_TIER_ENV} (rounded down to whole GiB per rank)")
    s.add_argument("--context", metavar="SIZE", help="context in tokens or binary k/m (512k = 524288); "
                   "auto-YaRN past native, checked against the resolved KV pool; explicit --hf-overrides wins")
    s.add_argument("--swarm", action="store_true", help="the row's tested high-concurrency cap instead of its default (experimental): "
                   "the Flash-Next 4 x 3090 profiles 24 sessions (default 8), the 27B 4 x 3090 (TP4) profiles 32 (default 24). Needs the KV RAM cache "
                   "at the size that cap was qualified at (48g per server): without --kv-ram-cache the cache is set to that size, a smaller "
                   "--kv-ram-cache is refused. Refused on rows with no swarm cap (the error names the rows that have one). An exported SEQS still "
                   "wins over every cap; the plan and the boot log say which cap applies (default, --swarm or SEQS)")
    s.add_argument("--dry-run", action="store_true", help="print the resolved command and env; start nothing")
    s.add_argument("--simulate", metavar="SPEC", help="preflight's hardware spec, e.g. 4x3090 or 8x4090+8x4090-48g:numa=2 (--dry-run only)")
    ls = sub.add_parser("list", help="checkpoint -> profile -> status -> hardware -> launcher table")
    ls.add_argument("checkpoint", nargs="?", help="only this checkpoint's rows")
    ls.add_argument("--arch", help="only rows for this arch (sm_80, sm_86, sm_89)")
    ls.add_argument("--cards", type=int, help="only rows of this card count")
    ls.add_argument("--vram", type=int, help="only rows of this VRAM class (GB)")
    sub.add_parser("doctor", help="run tools/merlin/preflight.py (all following args pass through); --report [FILE] writes a "
                   "redacted bug-report file (preflight, the serve plan, the server log's last lines)")
    sub.add_parser("pull", help="download and verify a checkpoint: merlin pull <checkpoint> [--models-dir DIR] [--ple int6] "
                   "(int6: also the family's PLE pack and the overlay DIR-int6)")
    v = sub.add_parser("verify", help="stream SHA-256 over every listed checkpoint file, including shards")
    v.add_argument("directory", help="checkpoint directory to verify (always rehashes)")
    v.add_argument("--checkpoint", choices=list(CHECKPOINTS), help="recipe name when directory identification is ambiguous")
    v.add_argument("--ple", choices=("fp8", "int6"), default="fp8", help="checkpoint PLE table (default: fp8); int6: the "
                   "directory is an INT6 overlay, and its FP8 base and the pack are verified with it")
    args = ap.parse_args(argv)
    if args.cmd == "verify":
        if extra:
            ap.error("verify does not accept extra vllm arguments")
        return cmd_verify(args, out)
    if args.cmd == "list":
        try:
            return cmd_list(out, REPO, args.checkpoint, args.arch, args.cards, args.vram)
        except CliError as e:
            print(f"merlin: {e}", file=sys.stderr)
            return e.code
    return cmd_serve(args, extra, out)


if __name__ == "__main__":
    R.safe_stdio()
    sys.exit(main())

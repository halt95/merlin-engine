#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The one Merlin launcher: (checkpoint, profile, env overrides) -> ``exec vllm serve``.

    python3 tools/merlin/launch.py --profile <id> [--kv fp8|bf16] [--checkpoint <name>] -- <checkpoint-dir> [extra vllm args]

Every ``models/<m>/serve*.sh`` and every file under ``tools/merlin/launchers/`` is a shim that calls this module with
its profile id; ``merlin serve`` resolves the profile from the hardware and calls it too. The per-profile values come
from ``tools/merlin/profiles/*.json`` (registry.py); the family argv shapes (Flash-Next, 27B, Super VL, Ornith) and the
shared start-up logic (toolchain and CUDA_HOME default, build-product check, PLE readiness, image cap, topology
selection) live here, once.

Standard library only (Python >= 3.10); no GPU needed until ``vllm`` starts. ``SERVE_PRINT_ENV_AND_EXIT=1`` prints
the final platform decision and the vllm flags instead of starting the server (test hook).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import runpy
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# checkpoint name -> (registry model, recipe directory)
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "flash-next": ("flash-next", "models/flash-next"),
    "swift1.5": ("flash-next", "models/swift1.5"),
    "swift1.5-huihui": ("flash-next", "models/swift1.5-huihui"),
    "27b": ("27b", "models/qwen38-27b"),
    "27b-thinkingcap": ("27b", "models/thinkingcap"),
    "nemotron-super-vl": ("svl", "models/nemotron-super-vl"),
    "ornith1.5": ("ornith", "models/ornith1.5"),
}
FAMILY = {"flash-next": "flash-next", "27b": "27b", "svl": "svl", "ornith": "ornith"}

# Family constants: values no profile or checkpoint has ever varied (the engine's CLI contract per family).
FN_GENERATION = '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
CKPT_SCALES = "qsa_kv_scales.json"  # Flash-Next family: a checkpoint's own FP8 KV scales, next to its weights
VERIFIED_MARKER = ".merlin-verified.json"  # written by merlin setup after hashing every listed file (setup_cli)
FN_CHAT_TEMPLATE ="models/qwen38-27b/templates/froggeric-qwen-v22.5.jinja"  # shared with 27B / ThinkingCap
FN_KERNEL_CONFIG = '{"qsa_warmup_use_serving_bounds":true,"reuse_gdn_prefill_warmup":true,"persist_triton_autotune":true}'
FN_UNSET = (
    "VLLM_PLE_CPU_OFFLOAD VLLM_PLE_TEST_FAULT VLLM_PLE_TEST_FAULT_GATE VLLM_PLE_TEST_FAULT_LOG "
    "VLLM_QSA_KV_CLIP_COUNT VLLM_E2_MEMSTATS VLLM_E2_MEMSTATS_DIR VLLM_PLE_PROGRESS_DEADLINE_S PYTORCH_CUDA_ALLOC_CONF"
).split()
B27_INDUCTOR = '{"test_configs.force_filter_reduction_configs":true,"dynamic_scale_rblock":false}'
B27_GENERATION = FN_GENERATION
# Ornith 1.5: sampling comes from the checkpoint's generation_config (T 0.6 / top_p 0.95 / top_k 20); only the penalties are set
ORN_GENERATION = '{"presence_penalty":0.0,"repetition_penalty":1.0}'
ORN_KERNEL_CONFIG = '{"reuse_gdn_prefill_warmup":true,"persist_triton_autotune":true}'
# Flash-Next host memory: the ~48 GiB n-gram table + ~4.2 GiB pinned embeddings per server.
FN_RAM_QUALIFIED_KB = 96468992  # 92 GiB MemTotal floor for the qualified 96 GB class, allowing kernel reservations
FN_RAM_FLOOR_KB = 72351744  # ~69 GiB resident floor (its PLE-table term is the FP8 one: preflight.FN_PLE_TABLE_GIB)
# Measured on the reference host under a memory cap (systemd scope memory.max, swap 0, KV RAM cache off):
# 64 GiB OOM-killed at boot, 72 and 76 GiB booted and passed 262K identity + needle, so the floor lies in (64, 72];
# 72 GiB is the tested minimum.
FN_RAM_TESTED_MIN_GIB = 72
# The FP8 table term inside FN_RAM_FLOOR_KB (preflight.FN_PLE_TABLE_GIB): a checkpoint with a PLE_OPT_IN_TABLES table
# (INT6 g32, --ple int6) swaps it for that table's size; the floor and its other terms are unchanged.
FN_PLE_FP8_GIB = 47.68
FN_SHM_MIN_KB = 1048576
FN_KV_HOST_TIER = "MERLIN_KV_HOST_TIER_GIB"  # host-RAM KV tier, GiB per rank (engine knob; merlin serve --kv-ram-cache sets it)
# The KV RAM cache ships as experimental in 3.0.1 on the Flash-Next family.
FN_KV_TIER_EXPERIMENTAL = ("WARNING: KV RAM cache on Flash-Next is experimental in 3.0.1: a session evicted under heavy "
                           "concurrent prefill may recompute (output identical)")
# The 27B KV RAM cache (TP1 / TP2 / TP4, ThinkingCap, no-P2P twins) is experimental in 3.0.1 as well.
KV_TIER_EXPERIMENTAL_27B = ("WARNING: KV RAM cache on the 27B is experimental in 3.0.1: a session evicted under heavy "
                            "concurrent prefill may recompute (output identical)")
FN_MARLIN_MOE_CFG = "MERLIN_MARLIN_MOE_CFG"  # per-shape Marlin MoE kernel choice; the engine parses it (vllm/merlin/marlin_moe_cfg.py)
# DEBUG ONLY host-tier fault injection: fail the first N restore / demotion transfers per rank so a fault-injection
# test can drive the failed-transfer path (vllm/merlin/kv_host_tier_faults.py). Never for serving: passed only when nonzero and
# only with the tier on, with a loud note.
FN_KV_HOST_TIER_FAULTS = ("MERLIN_KV_HOST_TIER_FAIL_LOADS", "MERLIN_KV_HOST_TIER_FAIL_STORES")
# The host-tier pacing knobs and the debug census (vllm/merlin/kv_host_tier_writeback.GATE_ENVS + the census switch);
# passed through with the tier on only. The engine validates the integers; the census takes exactly 0 or 1. The
# admission flag (kv_host_tier_policy.admission_mode) takes a mode name, default off.
FN_KV_HOST_TIER_ADMISSION = "MERLIN_KV_HOST_TIER__ADMISSION"
FN_KV_HOST_TIER_ADMISSION_MODES = ("0", "off", "lru", "slru")
FN_KV_HOST_TIER_GATES = ("MERLIN_KV_HOST_TIER__COPY_CREDITS", "MERLIN_KV_HOST_TIER__ESCAPE_STEPS",
                         "MERLIN_KV_HOST_TIER__N_RESTORE", "MERLIN_KV_HOST_TIER__MAX_ASSEMBLING",
                         FN_KV_HOST_TIER_ADMISSION, "MERLIN_KV_HOST_TIER__COPY_CREDITS_D",
                         "MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB")
FN_KV_HOST_TIER_CENSUS = "MERLIN_KV_HOST_TIER_CENSUS"
FN_WORKLOAD_TILE = "MERLIN_FLASHINFER_WORKLOAD_TILE"  # the opt-in FlashInfer prefill tile; not used on Flash-Next
# Every knob that is refused unless the tier is on: a layout pass that plans with the tier off leaves them out.
FN_KV_HOST_TIER_NEEDS_ON = (*FN_KV_HOST_TIER_FAULTS, *FN_KV_HOST_TIER_GATES, FN_KV_HOST_TIER_CENSUS)
# Every family: an SSE comment line (": keep-alive") every 15 s while a stream sends
# nothing (queued, a long prefill, between tokens), under the 30-60 s read timeouts of
# common proxies. The engine's own default is 0 (off); a
# trailing "--sse-keep-alive-interval 0" turns it off again (extra args come last, win).
SSE_KEEP_ALIVE_S = 15
HOOK_VARS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK", "VLLM_PP_LAYER_PARTITION",
    "NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_PROTO", "NCCL_ALGO", "VLLM_QSA_KV_SCALES",
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", "VLLM_WSL2_ENABLE_PIN_MEMORY",
)


class LaunchError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


@dataclass
class Plan:
    model: str
    checkpoint: str  # checkpoint name
    profile: dict[str, Any]
    ckpt: str  # checkpoint directory
    env: dict[str, str]
    args: list[str]  # the vllm flags between the checkpoint and the extra args
    extra: list[str]
    repo: Path
    venv: str
    tree: str | None
    cache_root: str
    status: str
    p2p_mode: str | None = None
    prefix: list[str] = field(default_factory=list)  # topology launch prefix (numactl ...), selector on only
    notes: list[str] = field(default_factory=list)  # stderr notes produced while resolving defaults
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, refusal message) checked before exec
    scales_recipe: str | None = None  # Flash-Next: the checkpoint name whose recipe FP8 KV scales are in use
    topo: dict[str, Any] = field(default_factory=dict)  # Flash-Next: AUTO_TOPO decision inputs
    caller_env: dict[str, str] = field(default_factory=dict)  # the environment as the caller gave it

    @property
    def family(self) -> str:
        return FAMILY[self.model]

    def argv(self) -> list[str]:
        return [*self.prefix, f"{self.venv}/bin/vllm", "serve", self.ckpt, *self.args, *self.extra]


# PLE table formats besides the shipped FP8 (text_config.ple_embedding_dtype -> table GiB in the offload process).
# INT6 g32 is the `--ple int6` opt-in table (the route for smaller hosts): a checkpoint carrying it boots without a
# test switch. merlin serve / setup refuse --ple int6 where its pack is not available (setup_cli.int6_reason). The
# BF16 reference and INT8-row tables are test tables, never shipped: launch refuses them unless
# MERLIN_PLE_TEST_TABLE names the checkpoint's exact format.
PLE_OPT_IN_TABLES = {"int6_g32": 38.74}
PLE_TEST_TABLES = {"bfloat16": 95.37, "int8_row": 48.28}


def ple_table_gib(ple: str | None) -> float:
    """Host RAM of the checkpoint's PLE table (``ple_embedding_dtype``), the figure the RAM floor counts: the opt-in
    and test tables at their own size, the FP8 default otherwise."""
    return {**PLE_OPT_IN_TABLES, **PLE_TEST_TABLES}.get(ple or "", FN_PLE_FP8_GIB)
# Flash-Next-family checkpoints whose FP8 PLE shards are the INT6 pack's but whose INT6 boot failed a quality gate:
# refused with the reason by merlin serve / setup / pull (setup_cli.int6_reason) and here, so a recipe shim or a
# hand-built overlay cannot boot one either.
PLE_INT6_UNQUALIFIED = {"swift1.5-huihui": "its MTP acceptance with the INT6 table falls below the qualification bar"}


def int6_unqualified_reason(checkpoint: str | None) -> str | None:
    """The refusal for an INT6 PLE table on ``checkpoint``, or None when INT6 is not refused for it."""
    if checkpoint in PLE_INT6_UNQUALIFIED:
        return f"INT6 PLE is not qualified for {checkpoint}: {PLE_INT6_UNQUALIFIED[checkpoint]}; serve it with --ple fp8"
    return None
# The reduced MTP draft vocabulary (REDUCED_DRAFT_VOCAB / DRAFT_VOCAB) is gone: a value in the environment is refused
# rather than ignored, so a stale shell cannot believe it is serving a reduced head.
REDUCED_DRAFT_VOCAB_REMOVED = "reduced draft vocabulary was removed in Merlin 3.0.1; the MTP drafter uses the full vocabulary"

RETIRED_KNOBS = {  # read by nothing; a value in the environment is a stale shell, say so
    "VLLM_E2_GUARD_IDLE": "guard idle parking is built in (v2.5.2)",
    "VLLM_E2_PLE_PULL_TRANSPORT": "the pull transport is the only PLE transport",
    "VLLM_E2_SIDE_CACHE_LAYOUT": "the side-cache layout is always on",
    "VLLM_E2_MAMBA_RETAIN_CHECKPOINT": "checkpoint retention is always on",
    "VLLM_MROPE_CACHE_1X": "the 1x M-RoPE cache is automatic for images",
    "VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK": "removed in Merlin 3.0.1; it was off by default and measured slower",
}
# Knobs a user's shell may carry that the Flash-Next profile sets itself (or leaves at the engine default):
# dropped on top of MERLIN_* and FN_UNSET so a stale export cannot change a qualified profile.
FN_RESET_EXTRA = ("VLLM_DRAFT_BOUNDED_PLAN", "VLLM_MERLIN_VISION_PEAK", "VLLM_GDN_BA_AUX_STREAM", "VLLM_E2_GUARD_TRANSPORT")


def distributed_ports(port: str) -> tuple[int, int]:
    """MASTER_PORT / VLLM_PORT for one server from its HTTP PORT (vLLM also takes VLLM_PORT + 1).

    PORT 8000..8355 maps to 29500+d / 30000+100d (the values the shipped profiles use). Any other PORT p, with r = p % 10000:
    below 20000 -> 20000+r / 40000+2r; 20000..39999 -> 10000+r / 40000+2r; 40000 and up -> 20000+r / 30000+2(p % 5000).
    Each band lies outside the HTTP range it serves, so the three ports never meet the HTTP port, and consecutive
    PORTs (``merlin serve --instances``) get disjoint sets; all stay below 65536.
    """
    try:
        p = int(port)
    except ValueError:
        raise LaunchError(f"PORT must be a whole number (got '{port}')", 2) from None
    if not 1024 <= p <= 65535:
        raise LaunchError(f"PORT must be between 1024 and 65535 (got {p})", 2)
    d, r = p - 8000, p % 10000
    if 0 <= d < 356:
        return 29500 + d, 30000 + d * 100
    if p < 20000:
        return 20000 + r, 40000 + 2 * r
    if p < 40000:
        return 10000 + r, 40000 + 2 * r
    return 20000 + r, 30000 + 2 * (p % 5000)


# Gate knobs the engine takes as 0 = rows only (``_e2_nonneg_int``); every other whole-number gate is a positive count
# (``_e2_positive_int``), so 0 is refused here as the engine would at boot.
FN_KV_HOST_TIER_GATES_NONNEG = ("MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB",)


def _kv_tier_gates(environ_in: dict[str, str], env: dict[str, str]) -> None:
    """The host-tier pacing knobs, the same on every family: blank = unset (not exported), a value is
    checked as the engine checks it (the admission mode name; positive whole numbers, the byte cap may be 0) so a typo
    fails before the boot, and needs the tier on (``FN_KV_HOST_TIER`` already resolved into ``env``)."""
    for name in FN_KV_HOST_TIER_GATES:
        n = (environ_in.get(name) or "").strip()
        if not n:
            env.pop(name, None)
            continue
        if name == FN_KV_HOST_TIER_ADMISSION:
            if n.lower() not in FN_KV_HOST_TIER_ADMISSION_MODES:
                raise LaunchError(f"{name} must be one of {', '.join(FN_KV_HOST_TIER_ADMISSION_MODES)} (got {n!r})", 1)
        elif name in FN_KV_HOST_TIER_GATES_NONNEG:
            _whole_knob(name, n, n.isascii() and n.isdigit(), "a non-negative whole number")
        elif not _whole_knob(name, n, n.isascii() and n.isdigit(), "a positive whole number"):
            raise LaunchError(f"{name} must be a positive whole number, got {n!r}", 1)
        if FN_KV_HOST_TIER not in env:
            raise LaunchError(f"{name}={n} needs the host-RAM KV tier on ({FN_KV_HOST_TIER} > 0)", 1)
        env[name] = n


def _restore_unit_gate(model: str, p: dict[str, Any], tier_gib: int) -> None:
    """Refuse a KV RAM cache too small for one 262K restore unit on this row before anything loads (the engine would
    refuse it at init, after the weights are in)."""
    ranks = int(p["match"]["cards"])
    minimum = R.restore_unit_min_gib_per_rank(model, p, ranks)
    if tier_gib < minimum:
        raise LaunchError(f"{FN_KV_HOST_TIER}={tier_gib} GiB per rank cannot hold one 262K restore unit on {p['id']}; it "
                          f"needs at least {minimum} GiB per rank (merlin serve --kv-ram-cache {minimum * ranks}g). "
                          f"Increase {FN_KV_HOST_TIER} or set it to 0 to disable the KV RAM cache.", 1)


def _kv_tier_census(environ_in: dict[str, str], env: dict[str, str]) -> str | None:
    """The debug census, the same on every family: 0 or blank is not exported, 1 needs the tier on (``FN_KV_HOST_TIER`` already
    resolved into ``env``), anything else is refused. Returns the note for an active census."""
    census = (environ_in.get(FN_KV_HOST_TIER_CENSUS) or "").strip()
    if census not in ("", "0", "1"):
        raise LaunchError(f"{FN_KV_HOST_TIER_CENSUS} must be 0 or 1 (got {census!r})", 1)
    if census != "1":
        env.pop(FN_KV_HOST_TIER_CENSUS, None)
        return None
    if FN_KV_HOST_TIER not in env:
        raise LaunchError(f"{FN_KV_HOST_TIER_CENSUS}=1 needs the host-RAM KV tier on ({FN_KV_HOST_TIER} > 0)", 1)
    env[FN_KV_HOST_TIER_CENSUS] = "1"
    return f"note: {FN_KV_HOST_TIER_CENSUS}=1: the tier census (debug bookkeeping and log lines) is on"


def warn_retired(env: dict[str, str]) -> None:
    for k, why in RETIRED_KNOBS.items():
        if k in env:
            print(f"note: {k}={env[k]!r} is a retired knob and is ignored ({why})", file=sys.stderr)


def _knob(env: dict[str, str], name: str, default: str, assign: bool, keep_empty: bool = False) -> str:
    """Read an env knob with bash's semantics: ``assign`` = ``NAME=${NAME:-default}`` (an exported variable keeps its
    export with the resolved value), else inline ``${NAME:-default}`` (environment untouched); ``keep_empty`` =
    ``${NAME-default}`` (set-but-empty stays empty)."""
    present = name in env
    raw = env.get(name)
    if raw is None or (raw == "" and not keep_empty):
        val = default
    else:
        val = raw
    if assign and present:
        env[name] = val
    return val


def _digits(v: str) -> bool:
    """ASCII decimal digits only: superscript two (U+00B2) and other Unicode digits pass str.isdigit() but int()
    refuses them."""
    return v != "" and v.isascii() and v.isdigit()


def _small_whole(v: str) -> int | None:
    """``v`` as an int when it is ASCII digits that int() converts (not over Python's digit limit), else None."""
    if not _digits(v):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _whole_knob(name: str, raw: str, well_formed: bool, what: str) -> int:
    """A stripped whole-number knob: 0 when blank. A malformed value, or one int() refuses (over Python's
    4300-digit limit, or a digit character it cannot convert), is a LaunchError, never a traceback."""
    if not raw:
        return 0
    try:
        if not well_formed:
            raise ValueError
        return int(raw)
    except ValueError:
        shown = raw if len(raw) <= 40 else f"{raw[:20]}...({len(raw)} characters)"
        raise LaunchError(f"{name} must be {what}, got {shown!r}", 1) from None


# --------------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------------
def load_profile(
    model: str, profile_id: str, kv: str | None = None, kv_dtype: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reg = R.with_kv_dtype(R.load_registry(model), kv_dtype)
    if kv_dtype is not None and kv == "bf16":
        raise R.ProfileError("KV_DTYPE=fp8 conflicts with --kv bf16; unset KV_DTYPE for BF16")
    p = R.get_profile(reg, profile_id)
    if kv:
        dtype = {v: k for k, v in R.KV_TAG.items()}[kv]
        p = R.variant_profile(p, dtype)
    return reg, p


def default_checkpoint(model: str, profile: dict[str, Any]) -> str:
    if model == "flash-next":
        return profile.get("checkpoint") or "flash-next"
    return {"27b": "27b", "svl": "nemotron-super-vl", "ornith": "ornith1.5"}[model]


def build(
    model: str,
    profile_id: str,
    ckpt: str,
    environ: dict[str, str],
    extra: list[str] | None = None,
    kv: str | None = None,
    checkpoint: str | None = None,
    repo: Path | None = None,
    status: str | None = None,
) -> Plan:
    """Resolve env + argv for one profile from the registry and the caller's environment. ``status`` is the label
    merlin serve evaluated on the actual cards (``untested`` for a tested row on other names or without P2P); the
    registry row's own status otherwise.

    Pure apart from the directory-existence reads behind the Flash-Next VENV / CACHE_ROOT fallbacks; nothing is
    executed and nothing is written. The host checks (tree, venv, toolchain, topology, files) run in ``run``.
    """
    repo = Path(repo or REPO)
    try:
        reg, p = load_profile(model, profile_id, kv, environ.get("KV_DTYPE"))
    except R.ProfileError as e:
        raise LaunchError(str(e), 2) from None
    checkpoint = checkpoint or default_checkpoint(model, p)
    if checkpoint not in CHECKPOINTS or CHECKPOINTS[checkpoint][0] != model:
        raise LaunchError(f"checkpoint {checkpoint!r} does not belong to the {model} registry", 2)
    if model == "flash-next" and (p.get("checkpoint") or "flash-next") != checkpoint:
        raise LaunchError(f"profile {profile_id} serves {p.get('checkpoint') or 'flash-next'}, not {checkpoint}", 2)
    env = dict(environ)
    extra = list(extra or [])
    if any(a.split("=", 1)[0].replace("_", "-") == "--skip-tokenizer-init" for a in extra):
        raise LaunchError("--skip-tokenizer-init is unsupported by Merlin: tokenizer initialization is required "
                          "to validate prompt token IDs; remove this flag from the extra arguments", 2)
    # --swarm (SWARM=1): checked before any builder runs, on every family; sets the KV RAM cache when unset
    swarm_note = _swarm_prepare(p, env) if swarm_requested(env) else None
    if model == "flash-next":
        plan = _build_flash_next(reg, p, checkpoint, ckpt, env, extra, repo)
    elif model == "27b":
        plan = _build_27b(reg, p, checkpoint, ckpt, env, extra, repo)
    elif model == "ornith":
        plan = _build_ornith(reg, p, checkpoint, ckpt, env, extra, repo)
    else:
        plan = _build_svl(reg, p, checkpoint, ckpt, env, extra, repo)
    warning = R.sm80_fp8_warning(model, p)
    if warning:
        plan.notes.append(warning)
    if model != "flash-next" and "GUARD" in environ:
        plan.notes.append(
            f"GUARD={environ['GUARD']!r} is ignored on the {model} lane; GUARD is only honoured on Flash-Next"
        )
    if swarm_note:
        plan.notes.append(swarm_note)
    plan.env.pop(SWARM_ENV, None)  # a launcher knob, not the engine's
    _merge_speculative_config(plan)
    plan.args += ["--sse-keep-alive-interval", str(SSE_KEEP_ALIVE_S)]
    _note_extra_overrides(plan)
    shown = (R.LABEL_TESTED, R.LABEL_TESTING, R.LABEL_UNTESTED)
    if status in shown and not p.get("fp8_opt_in"):
        plan.status = R.label_text(reg, p, status)
    return plan


def _speculative_config_arg(arg: str) -> tuple[str, str | None] | None:
    """``(dotted key path, inline value or None)`` when ``arg`` is a spelling vLLM reads as --speculative-config
    (underscores for dashes, the -sc short form, ``=value``, and the dotted ``--speculative-config.KEY value`` form,
    where the path is ``KEY``); None for any other argument."""
    name, eq, value = arg.partition("=")
    if name.startswith("--"):
        head, dot, tail = name.partition(".")
        name = head.replace("_", "-") + dot + tail
    base, _dot, path = name.partition(".")
    if base == "-sc" or base == "--speculative-config":
        return path, (value if eq else None)
    if not eq and arg.startswith("-sc{"):  # the short form with its JSON attached
        return "", arg[3:]
    return None


_HUMAN_INT = re.compile(r"(\d+(?:\.\d+)?)([kKmMgGtT])")
_HUMAN_DECIMAL = {"k": 10**3, "m": 10**6, "g": 10**9, "t": 10**12}
_HUMAN_BINARY = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}


def _dotted_value(raw: str) -> Any:
    """A dotted ``--speculative-config.KEY`` value as vLLM reads it (vllm/utils/argparse_utils.py): JSON when it
    parses, else a human-readable integer (``256K`` = 262144, ``25.6k`` = 25600; no decimals with a binary suffix),
    else the plain string."""
    try:
        return json.loads(raw)
    except ValueError:
        pass
    m = _HUMAN_INT.fullmatch(raw.strip())
    if m:
        number, suffix = m.groups()
        if suffix in _HUMAN_DECIMAL:
            return int(float(number) * _HUMAN_DECIMAL[suffix])
        if "." not in number:
            return int(number) * _HUMAN_BINARY[suffix]
        return raw
    try:
        return int(raw.strip())
    except ValueError:
        return raw


def _merge_speculative_config(plan: Plan) -> None:
    """Merge extra speculative settings into the recipe's JSON, retaining the recipe's method, K and omitted draft
    fields. Every spelling vLLM accepts is merged and checked: --speculative-config / --speculative_config / -sc with
    a JSON object (as ``X value`` or ``X=value``), and the dotted ``--speculative-config.KEY value`` form, which vLLM
    applies after the JSON forms."""
    flag = "--speculative-config"
    if flag not in plan.args or not any(_speculative_config_arg(a) is not None for a in plan.extra):
        return
    pos = plan.args.index(flag) + 1
    spec = json.loads(plan.args[pos])
    extra: list[str] = []
    objects: list[str] = []
    dotted: dict[str, Any] = {}
    i = 0
    while i < len(plan.extra):
        arg = plan.extra[i]
        i += 1
        parsed = _speculative_config_arg(arg)
        if parsed is None:
            extra.append(arg)
            continue
        path, raw = parsed
        if raw is None:
            raw = plan.extra[i] if i < len(plan.extra) else ""
            i += 1
        if not path:
            objects.append(raw)
            continue
        if path.endswith("+"):  # vLLM's list form: comma-separated strings
            path, value = path[:-1], raw.split(",")
        else:
            value = _dotted_value(raw)
        keys = path.split(".")
        if not all(keys):
            raise LaunchError(f"{arg}: not a valid {flag}.KEY argument; use SPEC_K / --spec for launcher settings", 2)
        node = dotted
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
    if dotted:
        objects.append(json.dumps(dotted))
    for raw in objects:
        try:
            user = json.loads(raw)
            if not isinstance(user, dict):
                raise ValueError("expected a JSON object")
            json.dumps(user, allow_nan=False)  # json.loads also accepts non-JSON NaN / Infinity; refuse those.
        except ValueError as e:
            raise LaunchError(f"{flag} must be a valid JSON object ({e}); "
                              "use SPEC_K / --spec for launcher settings", 2) from None
        if "method" in user and user["method"] != spec["method"]:
            raise LaunchError(f"{flag} cannot change method from {spec['method']!r} to {user['method']!r}; "
                              "use SPEC_K / --spec for launcher settings", 2)
        if "num_speculative_tokens" in user and user["num_speculative_tokens"] != spec["num_speculative_tokens"]:
            raise LaunchError(
                f"{flag} cannot change num_speculative_tokens from {spec['num_speculative_tokens']!r} "
                f"to {user['num_speculative_tokens']!r}; use SPEC_K on configurable 27B profiles or the recipe's K "
                "so CUDA-graph capture sizes stay in sync", 2)
        for key, value in user.items():
            if key in spec and spec[key] != value:
                plan.notes.append(f"note: {flag} overrides {key}: {spec[key]!r} -> {value!r}")
        spec.update(user)
    plan.args[pos] = json.dumps(spec, separators=(",", ":"))
    plan.extra = extra


# vLLM flags a launcher knob owns: an extra copy after `--` wins (vLLM keeps the last) but skips the knob's checks.
KNOB_FOR_FLAG = {"--max-model-len": "MAXLEN", "--max-num-seqs": "SEQS", "--max-num-batched-tokens": "CHUNK",
                 "--long-prefill-token-threshold": "LPT"}
# vLLM's short spellings of flags the recipes set (vllm/engine/arg_utils.py).
SHORT_FLAGS = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size", "-dp": "--data-parallel-size",
               "-q": "--quantization"}


def _flag_values(args: list[str]) -> dict[str, str]:
    """``--flag value`` and ``--flag=value`` pairs under the name vLLM resolves them to (underscores read as dashes
    up to the first dot, the short spellings expanded); a bare ``--x`` / ``--no-x`` is x on / off. The last one wins,
    as in vLLM."""
    out: dict[str, str] = {}
    for i, arg in enumerate(args):
        flag, eq, value = arg.partition("=")
        if flag.startswith("--"):
            head, dot, tail = flag.partition(".")
            flag = head.replace("_", "-") + dot + tail
        elif flag in SHORT_FLAGS:
            flag = SHORT_FLAGS[flag]
        else:
            continue
        if eq:
            out[flag] = value
        elif i + 1 < len(args) and not (args[i + 1].startswith("--") or args[i + 1].partition("=")[0] in SHORT_FLAGS):
            out[flag] = args[i + 1]  # a value, possibly negative
        elif flag.startswith("--no-"):
            out["--" + flag[5:]] = "off"
        else:
            out[flag] = "on"
    return out


def _note_extra_overrides(plan: Plan) -> None:
    """Note every extra flag that overrides a value the recipe set. The launch is unchanged: an extra
    --max-model-len is how a YaRN launch is often written, so this informs rather than refuses."""
    recipe = _flag_values(plan.args)
    for flag, value in _flag_values(plan.extra).items():
        if flag in recipe and recipe[flag] != value:
            knob = KNOB_FOR_FLAG.get(flag)
            plan.notes.append(f"note: extra {flag} {value} overrides the recipe's {recipe[flag]}"
                              + (f"; {knob}={value} sets it with the launcher's checks" if knob else ""))


def _abs(path: str, cwd: str) -> str:
    return path if os.path.isabs(path) else f"{cwd}/{path}"


def _default_dir(notes: list[str], var: str, static: str, old: str, advice: str) -> str:
    if not os.path.exists(static) and os.path.exists(old):
        notes.append(f"note: {var} defaults to {static}, which does not exist; using the legacy directory {old} for now. The legacy names are")
        notes.append(f"      read through Merlin 3.0.x only: {advice}.")
        return old
    return static


def cache_root_for(
    model: str, env: dict[str, str], repo: Path | None = None, notes: list[str] | None = None,
) -> str:
    """The cache root the family will pass to vLLM; doctor uses the same defaults and override precedence."""
    if model == "27b" and env.get("VLLM_CACHE_ROOT"):  # an empty override is unset, not the filesystem root
        return env["VLLM_CACHE_ROOT"]
    if env.get("CACHE_ROOT"):
        return env["CACHE_ROOT"]
    root = (repo or REPO).as_posix()
    if model == "flash-next":
        return _default_dir(
            notes if notes is not None else [], "CACHE_ROOT", f"{root}/.vllm-cache", f"{root}/.vllm-cache-v2.5",
            "rename it to .vllm-cache (the cache is keyed by hash, so it carries over)")
    return f"{root}/.vllm-cache"


def check_cache_links(cache_root: str) -> None:
    """Refuse dangling cache links before workers load weights; missing directories and valid links are fine."""
    root = Path(os.path.expanduser(cache_root))
    for path in (*reversed(root.parents), root, root / "torch_compile_cache"):
        if path.is_symlink() and not path.exists():
            raise LaunchError(
                f"VLLM_CACHE_ROOT={cache_root}: dangling cache symlink {path} -> {os.readlink(path)}; "
                "restore the target or replace the symlink with a real directory before starting, or set CACHE_ROOT "
                "to a fresh directory (also unset VLLM_CACHE_ROOT if it was explicitly set)", 1)


def _img_cap(env: dict[str, str], default: int) -> str:
    v = env.get("IMG_CAP_PX") or str(default)
    if not _digits(v) or v.startswith("0"):
        raise LaunchError(f"IMG_CAP_PX must be a whole number of pixels, got '{v}'", 1)
    if len(v) > 10 or int(v) < 65536:
        raise LaunchError(f"IMG_CAP_PX must be between 65536 (the shortest_edge floor) and 9999999999 pixels, got {v}", 1)
    return v


def _max_pixels_27b(v: str) -> None:
    if v == "":
        return
    if v.startswith("0") or not _digits(v):
        raise LaunchError("MAX_PIXELS must be empty or a canonical decimal pixel count (no leading zero, digits only)", 2)
    if len(v) > 8 or int(v) < 65536 or int(v) > 33554432:
        raise LaunchError(f"MAX_PIXELS must be between 65536 and 33554432 (got {v})", 2)


# ---------------------------------------------------------------------------------------------------------------
# Session caps (these replace the earlier cache-gated raise and the unconditional 27B TP4 32):
# concurrency is decoupled from the KV RAM cache. Every row serves its 3.0.0 ``max_num_seqs`` whether or not the cache is
# on. A row that has a tested high-concurrency cap (``serve.kv_tier_seqs``: the cap, and the GiB per rank of KV RAM cache
# it was qualified with) gets it only when asked for: ``merlin serve --swarm`` / ``SWARM=1``. An exported SEQS wins.
# ---------------------------------------------------------------------------------------------------------------
SWARM_ENV = "SWARM"
CAP_NOTE_PREFIXES = ("note: concurrency ", "WARNING: concurrency ", "WARNING: SEQS=")  # what _session_cap writes


def is_cap_note(note: str) -> bool:
    return note.startswith(CAP_NOTE_PREFIXES)


UNTESTED_SWARM = "untested \u26ab (expected to work)"  # what the notes say about a row whose swarm cap no qualification run has covered


def swarm_tested(p: dict) -> bool:
    """True when a qualification run covered this row at its swarm cap (``kv_tier_seqs.status`` = tested); a gate without the field, or
    marked untested, is not: the notes then say so."""
    gate = swarm_gate(p)
    return bool(gate) and gate.get("status") == R.LABEL_TESTED


def swarm_gate(p: dict) -> dict | None:
    """The row's swarm cap block (``max_num_seqs``, ``min_gib_per_rank``), or None when the row has no qualified swarm cap."""
    return p["serve"].get("kv_tier_seqs") or None


def swarm_rows() -> list[tuple[str, int, int, int]]:
    """(profile id, swarm cap, GiB per rank of KV RAM cache it needs, ranks) of every row with a qualified swarm cap."""
    out = []
    for model in ("flash-next", "27b", "svl", "ornith"):
        for p in R.load_registry(model)["profiles"]:
            gate = swarm_gate(p)
            if gate:
                out.append((p["id"], int(gate["max_num_seqs"]), int(gate["min_gib_per_rank"]), int(p["match"]["cards"])))
    return out


def swarm_not_qualified(pid: str) -> str:
    """The refusal for --swarm on a row with no swarm cap, naming the rows that have one."""
    groups: dict[tuple[int, int], list[str]] = {}
    for r, cap, need, ranks in swarm_rows():
        groups.setdefault((cap, need * ranks), []).append(r)
    rows = "; ".join(f"{', '.join(ids)} ({cap} sessions with the KV RAM cache at {total}g)" for (cap, total), ids in groups.items())
    return (f"--swarm (SWARM=1): {pid} has no qualified swarm cap. Rows that have one: {rows}. Drop --swarm to serve "
            f"{pid} at its default cap, or choose one of those rows (--profile / --tp)")


def swarm_cache_refusal(p: dict, gib: int) -> str:
    """The refusal for --swarm with a KV RAM cache under the size the swarm cap was qualified with (never raised)."""
    gate, ranks = swarm_gate(p), int(p["match"]["cards"])
    cap, need = int(gate["max_num_seqs"]), int(gate["min_gib_per_rank"])
    have = f"{gib} GiB per rank = {gib * ranks}g on {ranks} ranks" if gib else "the cache off (0 GiB per rank)"
    return (f"--swarm (SWARM=1): the {cap}-session cap needs the KV RAM cache at {need} GiB per rank = "
            f"{need * ranks}g on {ranks} ranks (the size it was tested at), and this launch has {have}; the cache is not raised for you. Give "
            f"--kv-ram-cache {need * ranks}g or more ({FN_KV_HOST_TIER}={need} or more for a direct launch), or drop "
            f"--swarm: the default cap is {p['serve']['max_num_seqs']}")


def swarm_requested(env: dict[str, str]) -> bool:
    """SWARM=1 (merlin serve --swarm, or exported): the one swarm decision. Unset, blank or 0 is off; anything else is refused."""
    v = (env.get(SWARM_ENV) or "").strip()
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise LaunchError(f"SWARM must be 1 (merlin serve --swarm) or 0, got {v!r}", 2)


def _tier_gib(environ_in: dict[str, str]) -> int:
    """The KV RAM cache GiB per rank as exported. A malformed value is refused by the tier check itself; here it is 0."""
    raw = (environ_in.get(FN_KV_HOST_TIER) or "").strip()
    return int(raw) if raw.isascii() and raw.isdigit() and len(raw) < 6 else 0


def _swarm_prepare(p: dict, env: dict[str, str]) -> str | None:
    """--swarm on this row: refused when the row has no swarm cap or the KV RAM cache is under the size the cap was
    qualified with (it is never raised for the user); with the cache unset it is set to that size (``env`` is updated,
    the note returned). The builders then read the cache from ``env`` like any exported one."""
    gate = swarm_gate(p)
    if not gate:
        raise LaunchError(swarm_not_qualified(p["id"]), 2)
    if (env.get("SPEC") or "").strip() == "dflash2":
        raise LaunchError("--swarm (SWARM=1) with SPEC=dflash2 (--spec dflash2): the KV RAM cache the swarm cap needs is qualified "
                          "with MTP only in 3.0.1. Drop --swarm (or unset SWARM), or use --spec mtp (the default)", 2)
    cap, need, ranks = int(gate["max_num_seqs"]), int(gate["min_gib_per_rank"]), int(p["match"]["cards"])
    raw = (env.get(FN_KV_HOST_TIER) or "").strip()
    if not raw:
        env[FN_KV_HOST_TIER] = str(need)
        return (f"note: --swarm: the KV RAM cache was not set, so it is {need} GiB per rank = {need * ranks}g on {ranks} "
                f"ranks, the size the {cap}-session cap was tested at (--kv-ram-cache SIZE sets another size, at least "
                f"{need * ranks}g)")
    if raw.isascii() and raw.isdigit() and len(raw) < 6 and int(raw) < need:
        raise LaunchError(swarm_cache_refusal(p, int(raw)), 2)
    return None


def cap_report(p: dict, swarm: bool, seqs: str, gib: int) -> tuple[str, list[str]]:
    """The session cap that applies on this row and the notes saying which and why (default / swarm / SEQS).
    ``seqs`` is the exported SEQS ("" when unset), ``gib`` the KV RAM cache in GiB per rank (0 = off). Pure; the
    launcher prints the notes at boot, ``merlin serve --dry-run`` and setup show them."""
    row = int(p["serve"]["max_num_seqs"])
    gate = swarm_gate(p)
    if not gate:
        return str(row), []
    cap, need, ranks = int(gate["max_num_seqs"]), int(gate["min_gib_per_rank"]), int(p["match"]["cards"])
    size = f"{need} GiB per rank = {need * ranks}g on {ranks} ranks"
    tested = swarm_tested(p)
    n = _small_whole(seqs.strip())
    if swarm:
        if n is not None:
            if n > cap:
                return str(cap), [f"WARNING: concurrency {n} (exported SEQS wins over --swarm's {cap}): above even the swarm "
                                  f"cap, the most that was tested; the default cap is {row}"]
            return str(cap), [f"note: concurrency {n} (exported SEQS wins over --swarm's {cap}; the default cap "
                              f"is {row})"]
        cache = (f"{gib} GiB per rank = {gib * ranks}g on {ranks} ranks, the size the cap was tested at" if gib == need
                 else f"{gib} GiB per rank = {gib * ranks}g on {ranks} ranks, above the {need * ranks}g the cap was tested at")
        return str(cap), [f"WARNING: concurrency {cap} (--swarm, experimental):{'' if tested else ' ' + UNTESTED_SWARM + ' on this row;'} "
                          f"the default cap is {row}; the KV RAM cache is {cache}"]
    if n is not None:
        if n > row:
            how = (f"{cap} is qualified only with --swarm (the KV RAM cache at {size} or more, experimental)" if tested else
                   f"{cap} is available only with --swarm ({UNTESTED_SWARM} on this row; the KV RAM cache at {size} or more, experimental)")
            return str(row), [f"WARNING: SEQS={n} is above the default cap {row}; {how}"]
        return str(row), []
    cache = f"; the KV RAM cache ({gib} GiB per rank) does not raise it" if gib else ""
    return str(row), [f"note: concurrency {row} (default{cache}); --swarm raises it to {cap} (experimental"
                      f"{'' if tested else '; ' + UNTESTED_SWARM + ' on this row'}; needs the KV RAM cache at {size} or more)"]


def _session_cap(p: dict, environ_in: dict[str, str], notes: list[str]) -> str:
    """The row's sequence cap: its 3.0.0 ``max_num_seqs``, or the swarm cap (``kv_tier_seqs``) under SWARM=1. The KV RAM
    cache never changes it. An exported SEQS wins either way (``_shape_knobs`` applies it); the notes say which cap
    applies and why."""
    cap, said = cap_report(p, swarm_requested(environ_in), environ_in.get("SEQS") or "", _tier_gib(environ_in))
    notes.extend(said)
    return cap


def _shape_knobs(env: dict[str, str], seqs_row: str, maxlen_row: str, code: int, notes: list[str]) -> tuple[str, str]:
    """SEQS / MAXLEN on a qualified Flash-Next or 27B row: unset -> the row's value; a value must be a whole number
    from 1, and one off the row says so (the row's KV pin was measured at its shape)."""
    seqs = _int_knob(env, "SEQS", True, seqs_row, code) or seqs_row
    maxlen = _int_knob(env, "MAXLEN", True, maxlen_row, code) or maxlen_row
    for name, v in (("SEQS", seqs), ("MAXLEN", maxlen)):
        if int(v) < 1:
            raise LaunchError(f"{name} must be a whole number from 1 (got '{v}')", code)
    if int(seqs) != int(seqs_row):
        notes.append(f"note: SEQS={seqs} is off the qualified row ({seqs_row}): its KV pin was measured at {seqs_row} "
                     "sequences; the CUDA-graph capture sizes follow SEQS")
    if int(maxlen) > int(maxlen_row):
        notes.append(f"note: MAXLEN={maxlen} is above the row's {maxlen_row}-token context: pass the matching rope/YaRN "
                     "override (--hf-overrides) as an extra argument, or vLLM refuses to start")
    elif int(maxlen) != int(maxlen_row):
        notes.append(f"note: MAXLEN={maxlen} is off the qualified row ({maxlen_row})")
    return seqs, maxlen


def _seqs_capture(sizes: list[int], seqs: str, seqs_row: str, k: int) -> list[int]:
    """CUDA-graph capture sizes for SEQS: the row's own list at its own SEQS, else the shipped ladder for SEQS x (k + 1)
    decode tokens (vLLM keeps an explicit list as given, so a row's list off its SEQS has dead or missing sizes)."""
    return list(sizes) if int(seqs) == int(seqs_row) else R.capture_sizes_for(int(seqs), k)


def _int_knob(env: dict[str, str], name: str, assign: bool, default: str | None, code: int) -> str | None:
    """CHUNK / BLOCK_SIZE / LPT style knobs: unset -> default (None = no flag); a value must be a whole number."""
    v = _knob(env, name, default or "", assign) if default is not None else env.get(name, "")
    if v == "":
        return None
    if _small_whole(v) is None:
        shown = v if len(v) <= 40 else f"{v[:20]}...({len(v)} characters)"
        raise LaunchError(f"{name} must be a whole number (got '{shown}')", code)
    return v


def _guard_mode(env: dict[str, str]) -> None:
    # GUARD takes precedence even when set empty (invalid); otherwise the inherited mode, then warn.
    if "GUARD" in env:
        mode = env["GUARD"]
    else:
        mode = env.get("VLLM_E2_GUARD_MODE", "warn")
    env["VLLM_E2_GUARD_MODE"] = mode
    if mode.strip().lower() not in ("raise", "warn"):
        raise LaunchError(f"VLLM_E2_GUARD_MODE must be raise|warn, got {mode.strip().lower()!r}", 1)
    interval = env.get("VLLM_E2_GUARD_INTERVAL")
    if interval is not None:
        try:
            value = float(interval)
        except ValueError:
            value = float("nan")
        if value != value or value in (float("inf"), float("-inf")) or value <= 0:
            raise LaunchError(f"VLLM_E2_GUARD_INTERVAL must be a finite number > 0, got {interval!r}", 1)


FN_FAMILY_NAMES = "flash-next, swift1.5, swift1.5-huihui"


def _spec_mtp_only(env: dict[str, str], who: str) -> None:
    """A checkpoint outside the Flash-Next family runs its own MTP head: SPEC=mtp (or unset) is that, anything else is
    refused instead of being ignored (3.0.1: DFlash2 is a Flash-Next-family drafter)."""
    spec = env.get("SPEC") or "mtp"
    if spec == "dflash2":
        raise LaunchError(f"DFlash2 is only available for the Flash-Next family ({FN_FAMILY_NAMES}); {who} uses its MTP head", 1)
    if spec != "mtp":
        raise LaunchError(f"SPEC must be mtp for {who} (its MTP head is its only drafter), got '{spec}' "
                          "(from the exported SPEC variable: unset it for the default)", 1)


def dflash2_rows(reg: dict) -> list[str]:
    """The profile rows with a qualified DFlash2 geometry (serve.dflash2: PP split, chunk, KV pin)."""
    return [q["id"] for q in reg["profiles"] if (q.get("serve") or {}).get("dflash2")]


def dflash2_override_warnings(profile: dict, env: dict[str, str]) -> list[str]:
    """Geometry overrides stay available for A/B runs, but carry no DFlash2 qualification."""
    dfl = profile["serve"].get("dflash2")
    if env.get("SPEC") != "dflash2" or not dfl:
        return []
    qualified = {"PP_PARTITION": ",".join(map(str, dfl["partition"])), "KV_BYTES": str(dfl["kv_cache_memory"])}
    return [
        f"SPEC=dflash2: {key}={env[key]} overrides qualified {key}={value}; untested geometry may fail the "
        "8 MP image floor or change the qualified token pool"
        for key, value in qualified.items() if env.get(key) and env[key] != value
    ]


def _flash_next_min_block(tp: int, fp8: bool, k: int) -> int:
    """Hybrid page floor for the pinned Flash-Next family, without importing the engine."""
    # qwen4_exp/nvidia/model.py: BF16 conv + FP32 GDN state / RecoverSSM record, sharded by TP;
    # the BF16 PLE conv is replicated. All shipped family checkpoints have these dimensions.
    gdn = (2 * 10240 * (3 + k) + 4 * 48 * 128 * 128) // tp
    if k:
        gdn += 4 * 48 * (k + 1) * 257 // tp
    ple = 2 * 10240 * (9 + k)
    # FullAttentionSpec: K + V, two KV heads (replicated above TP2), head_dim 256.
    token_bytes = 2 * max(1, 2 // tp) * 256 * (1 if fp8 else 2)
    # QSA's backend requires a multiple of 16 (platforms/interface.py:_align_hybrid_block_size).
    return 16 * ((max(gdn, ple) + 16 * token_bytes - 1) // (16 * token_bytes))


def _build_flash_next(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    environ_in = dict(env)
    base = dict(reg["base"])
    ck = reg.get("checkpoints", {}).get(p.get("checkpoint", "flash-next"), {})
    base.update(ck)
    lay, srv, ic = p["layout"], p["serve"], p["interconnect"]
    fp8 = srv["kv_cache_dtype"] == "fp8_e4m3"
    root = repo.as_posix()
    here = f"{root}/models/flash-next"  # the vanilla release assets (scales/, topo_select.py)
    cwd = os.getcwd()
    notes: list[str] = []
    # SPEC=dflash2 on a profile with a qualified DFlash2 geometry (serve.dflash2: PP split, chunk, KV pin) takes it as
    # its defaults; KV_BYTES / CHUNK / PP_PARTITION still override.
    dfl = (srv.get("dflash2") or None) if (env.get("SPEC") or "mtp") == "dflash2" else None
    if (env.get("SPEC") or "mtp") == "dflash2" and dfl is None:
        # the MTP pin is too high for the DFlash2 drafter (3.47e9 vs 4.1e9 B on 4 x 3090): no boot on the MTP geometry
        raise LaunchError(f"SPEC=dflash2: {p['id']} ({srv['kv_cache_dtype']} KV) has no "
                          f"qualified DFlash2 geometry (its KV pin is the MTP one); DFlash2 rows: {', '.join(dflash2_rows(reg))}. "
                          "Use SPEC=mtp (the default) here", 1)
    kv_bytes = _knob(env, "KV_BYTES", str((dfl or srv)["kv_cache_memory"]), assign=False)
    if not _digits(kv_bytes):
        raise LaunchError(f"KV_BYTES must be integer bytes (got {kv_bytes})", 2)
    tree = env.get("TREE") or root
    venv = env.get("VENV") or _default_dir(
        notes, "VENV", f"{root}/venv", f"{root}/venv-v2.5",
        f"rebuild with release/install-env.sh {root}/venv, which creates that environment")
    cache_root = cache_root_for("flash-next", env, repo, notes)
    tree, venv = _abs(tree, cwd), _abs(venv, cwd)
    for var, val in (("TREE", tree), ("VENV", venv)):
        if var in env:
            env[var] = val
    # pure knobs first: refuse bad settings before any host access
    img_cap_px = _img_cap(env, base["img_cap_px_default"])
    draft_head = env.get("DRAFT_HEAD") or R.DRAFT_HEAD_DEFAULT
    if draft_head not in ("bf16", "int8", "int4"):
        raise LaunchError(f"DRAFT_HEAD must be bf16, int8 or int4, got '{draft_head}'", 1)
    draft_spec = "" if draft_head == "bf16" else f',"draft_lm_head_quant":"{draft_head}"'
    if (env.get("REDUCED_DRAFT_VOCAB") or "0") != "0" or env.get("DRAFT_VOCAB"):
        raise LaunchError(f"{REDUCED_DRAFT_VOCAB_REMOVED} (unset REDUCED_DRAFT_VOCAB and DRAFT_VOCAB)", 1)
    # SPEC=dflash2: incoai's DFlash2 block-diffusion drafter (block 4 = 3 draft tokens,
    # the same verify width as MTP K=3) in place of the MTP head. BF16 draft and BF16
    # draft KV (an FP8 draft KV crashes on GDN hybrids). Opt-in; MTP stays the default.
    # SPEC=none: no drafter at all, a greedy reference for comparing drafters (never for serving).
    spec_method = env.get("SPEC") or "mtp"
    if spec_method not in ("mtp", "dflash2", "none"):
        raise LaunchError(f"SPEC must be mtp (default), dflash2 or none, got '{spec_method}' "
                          "(from the exported SPEC variable: unset it for the default)", 1)
    dflash2_model = None
    if spec_method == "dflash2":
        if env.get("DRAFT_HEAD"):
            raise LaunchError("DRAFT_HEAD applies to the MTP head only, not SPEC=dflash2", 1)
        dflash2_model = env.get("DFLASH2_MODEL")
        if not dflash2_model:
            raise LaunchError(
                "SPEC=dflash2 needs DFLASH2_MODEL=<local dir of incoai/Qwen3.8-Flash-Next-DFlash2>", 1)
        dflash2_model = _abs(dflash2_model, cwd)
    warm = env.get("WARM") or str(p["warm"])
    if warm not in ("0", "1"):
        raise LaunchError(f"WARM must be 1 (default) or 0, got '{warm}'", 1)
    chunk = _int_knob(env, "CHUNK", False, str((dfl or srv)["max_num_batched_tokens"]), 1)
    block = _int_knob(env, "BLOCK_SIZE", False, str(base.get("block_size", 4096)), 1)
    if env.get("BLOCK_SIZE"):
        minimum = _flash_next_min_block(lay["tp"], fp8, 0 if spec_method == "none" else R.MTP_K)
        if int(block) < minimum:
            # The floor comes from hand-copied checkpoint dimensions: a note, never a refusal.
            notes.append(f"note: BLOCK_SIZE={block} is below {minimum}, the estimated hybrid-state floor for Flash-Next "
                         f"({srv['kv_cache_dtype']} KV, TP{lay['tp']}, SPEC={spec_method}); the engine raises it to fit "
                         "if needed. Unset BLOCK_SIZE for the shipped default.")
    lpt = _int_knob(env, "LPT", False, None, 1)
    seqs, maxlen = _shape_knobs(env, _session_cap(p, environ_in, notes), str(base["max_model_len"]), 1, notes)
    # topology mode (the selector itself runs in the host phase)
    auto_topo = env.get("AUTO_TOPO", "0")
    if auto_topo in ("0", ""):
        topo_on = False
    elif auto_topo in ("1", "strict"):
        topo_on = True
    else:
        notes.append(f"topo: note: AUTO_TOPO='{auto_topo}' is not 0, 1 or strict; the selector stays off")
        topo_on = False

    env["PYTHONPATH"] = tree
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    # platform: GPUs, interconnect, layout
    modes = ic["modes"]
    switch = len(modes) > 1
    p2p_mode: str | None = None
    if switch:
        p2p_mode = env.get("P2P") or ic["default"]
        if p2p_mode not in modes:
            raise LaunchError(f"P2P must be on or off, got '{p2p_mode}'", 1)
    elif len(p["cuda_visible_devices"].split(",")) > 1:
        p2p_mode = next(iter(modes))
    mode = modes[p2p_mode] if p2p_mode else next(iter(modes.values()))
    if topo_on and not (p["match"]["cards"] == 4 and (p2p_mode or "on") == "on"):
        # the selector is for the four-card P2P-on layout; a no-P2P or non-quad row never used it
        notes.append(f"topo: note: AUTO_TOPO applies to four-card P2P-on rows only; ignored for {p['id']} (P2P {p2p_mode or 'n/a'})")
        topo_on = False

    def platform_env(e: dict[str, str]) -> None:
        e["CUDA_VISIBLE_DEVICES"] = e.get("CUDA_VISIBLE_DEVICES") or p["cuda_visible_devices"]
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        for var in mode.get("unset", []):
            e.pop(var, None)
        for k, v in mode.get("exports", {}).items():
            e[k] = str(v)

    platform_env(env)
    if lay["pp"] > 1:
        # PP_PARTITION: per-stage layer counts (e.g. 24,24) for a layout A/B; default = the profile's partition.
        partition = env.get("PP_PARTITION") or ",".join(str(x) for x in (dfl or lay)["partition"])
        parts = partition.split(",")
        if len(parts) != lay["pp"] or not all(_digits(x) and int(x) > 0 for x in parts) or (
                sum(map(int, parts)) != sum(lay["partition"])):
            raise LaunchError(f"PP_PARTITION must be {lay['pp']} positive layer counts summing to "
                              f"{sum(lay['partition'])} (got '{partition}')", 1)
        env["VLLM_PP_LAYER_PARTITION"] = partition
    else:
        env.pop("VLLM_PP_LAYER_PARTITION", None)
    tile = (environ_in.get(FN_WORKLOAD_TILE) or "").strip()
    if tile not in ("", "0"):  # stripped below with every other MERLIN_ export; say so rather than drop it silently
        notes.append(f"note: {FN_WORKLOAD_TILE}={tile!r} is ignored on Flash-Next; the Flash-Next profiles serve "
                     "with the FlashInfer tile they were qualified with")
    for v in [k for k in env if k.startswith("MERLIN_")] + FN_UNSET:
        env.pop(v, None)
    # The host-RAM KV tier size (GiB per rank, 0 = off; the engine reads it) is a deliberate MERLIN_ knob, set by
    # merlin serve --kv-ram-cache: it passes through, checked, and only when on.
    tier = (environ_in.get(FN_KV_HOST_TIER) or "").strip()
    tier_gib = _whole_knob(FN_KV_HOST_TIER, tier, tier.isdigit(), "a whole number of GiB per rank (0 = off)")
    if tier_gib:
        # 3.0.1 qualifies the tier with MTP only: with DFlash2 the host arena fills with draft-window pages and units are
        # lost to missed copies, so it pins host RAM for no restore.
        if spec_method == "dflash2":
            raise LaunchError(f"the KV RAM cache ({FN_KV_HOST_TIER}={tier_gib}, from merlin serve --kv-ram-cache, --swarm or an "
                              f"exported {FN_KV_HOST_TIER}) with SPEC=dflash2: the KV RAM cache is qualified with MTP only "
                              f"in 3.0.1; use SPEC=mtp (--spec mtp, the default) or turn the cache off (--kv-ram-cache 0, "
                              f"or {FN_KV_HOST_TIER}=0)", 2)
        _restore_unit_gate("flash-next", p, tier_gib)
        env[FN_KV_HOST_TIER] = str(tier_gib)
        notes.append(FN_KV_TIER_EXPERIMENTAL)
    # The Marlin MoE config is another deliberate MERLIN_ knob: passed verbatim when set, the engine validates it.
    moe_cfg = (environ_in.get(FN_MARLIN_MOE_CFG) or "").strip()
    if moe_cfg:
        env[FN_MARLIN_MOE_CFG] = moe_cfg
    # Debug-only exception: the host-tier fault-injection knobs pass through so a fault-injection test
    # reaches the engine. Same rule as the engine (plain non-negative integer; nonzero needs the tier
    # on), checked here so a typo fails before the boot; blank or 0 = off and not exported; a loud note when active.
    for name in FN_KV_HOST_TIER_FAULTS:
        n = (environ_in.get(name) or "").strip()
        quota = _whole_knob(name, n, n.isascii() and n.isdigit(),
                            "a non-negative whole number of transfers per rank (0 = off)")
        if quota:
            if FN_KV_HOST_TIER not in env:
                raise LaunchError(f"{name}={n} needs the host-RAM KV tier on ({FN_KV_HOST_TIER} > 0); it is a debug "
                                  "knob for the tier's failed-transfer path", 1)
            env[name] = str(quota)
            what = "restore" if name.endswith("_LOADS") else "demotion"
            notes.append(f"WARNING: DEBUG FAULT INJECTION ACTIVE: {name}={quota}: the first {quota} host-tier {what} "
                         "transfers on each rank will FAIL. Fault-injection testing only; never serve with it set.")
    # The tier's pacing knobs and debug census reach the engine like the tier size (an exception to the MERLIN_
    # strip above). Tier on only.
    _kv_tier_gates(environ_in, env)
    census_note = _kv_tier_census(environ_in, env)
    if census_note:
        notes.append(census_note)
    # Preserve scoring limits verbatim, including invalid values for the engine's positive-integer validation.
    for name in ("MERLIN_GENERATIVE_SCORING_MAX_ITEMS", "MERLIN_GENERATIVE_SCORING_MAX_TOKENS"):
        if name in environ_in:
            env[name] = environ_in[name]
    for v in FN_RESET_EXTRA:
        if v in env:
            notes.append(f"note: {v}={env[v]!r} dropped: the Flash-Next profile sets its own value (or the engine default) for it")
            env.pop(v)
    env["VLLM_PLE_OFFLOAD_READY_TIMEOUT"] = "1800"
    env["VLLM_PLE_OFFLOAD_HOME_DEVICE"] = env.get("PLE_HOME") or "0"
    env["VLLM_HOST_EMBED_TABLE"] = "1"
    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    files: list[tuple[str, str]] = []
    scales_recipe: str | None = None
    # Froggeric v22.5 on every Qwen profile: it accepts non-leading system messages, so /v1/messages
    # keeps a coding agent's per-turn notes in place (the checkpoint template refuses them and vLLM merges them into the
    # leading prompt, which breaks the prefix cache). CHAT_TEMPLATE overrides; empty serves the checkpoint's own template.
    tpl = _knob(env, "CHAT_TEMPLATE", f"{root}/{FN_CHAT_TEMPLATE}", True, keep_empty=True)
    template_args = ["--chat-template", tpl] if tpl else []
    if tpl:
        files.append((tpl, f"chat template missing: {tpl} (CHAT_TEMPLATE=<path>, or CHAT_TEMPLATE= for the checkpoint's own)"))
    if fp8:
        # SCALES wins; then the checkpoint's own sidecar (a variant ships its scales with its weights); then the recipe's
        own = Path(ckpt) / CKPT_SCALES
        if not env.get("SCALES") and own.is_file():
            notes.append(f"note: FP8 KV scales from the checkpoint: {own.as_posix()}")
        scales = env.get("SCALES") or (own.as_posix() if own.is_file() else
                                       f"{root}/{ck['scales']}" if ck else f"{here}/{base['scales']}")
        env["VLLM_QSA_KV_SCALES"] = scales
        # the recipe's scales were calibrated on its own weights: host_phase checks the directory is that checkpoint
        scales_recipe = None if env.get("SCALES") or own.is_file() else checkpoint
        files.append((scales, f"scales sidecar missing: {scales} (without it FP8 KV runs at scale 1.0)"))
    env["VLLM_CACHE_ROOT"] = cache_root
    warm_args: list[str] = []
    if warm == "1":
        env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
        warm_args = ["--kernel-config", FN_KERNEL_CONFIG]
    # no phone-home, like the other families: the vLLM usage reporter posts to stats.vllm.ai unless opted out
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    for k in ("MERLIN_FULL_K", "MERLIN_QSA_SORT", "MERLIN_TIE_DET", "MERLIN_TIE_RECENT", "MERLIN_TOPK_SORTED_EMIT"):
        env[k] = "1"
    _guard_mode(env)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_E2_PLE_FAST_IDS"] = "1"
    env["VLLM_E2_PLE_EARLY_REQUEST"] = "1"
    # One constant 256 MiB QSA prefill-logits block (granule = cap): exact-size per-chunk blocks ratchet the
    # allocator reserve towards the card wall at 262K.
    env["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "256"
    env["VLLM_E2_QSA_LOGITS_GRANULE_MB"] = "256"
    if env.get("COUNTERS", "0") == "1":
        env["MERLIN_TOPK_EXACT_COUNTERS_LOG"] = "30"
        env["VLLM_QSA_KV_CLIP_COUNT"] = "300"
    env.pop("VLLM_DRAFT_VOCAB_FILE", None)
    env.pop("VLLM_DRAFT_VOCAB_SHA256", None)
    names = (env.get("MODEL_NAME") or base["served_names"]).split()
    if spec_method == "none":
        spec = None
        notes.append("note: SPEC=none, no speculative decoding (identity reference only)")
    elif dflash2_model is not None:
        files.append((f"{dflash2_model}/config.json", f"DFlash2 draft missing: {dflash2_model} (DFLASH2_MODEL)"))
        spec = json.dumps({
            "method": "dflash", "model": dflash2_model, "num_speculative_tokens": R.MTP_K,
            "draft_sample_method": "probabilistic", "kv_cache_dtype": "auto",
        }, separators=(",", ":"))
        notes.append(f"note: SPEC=dflash2, draft {dflash2_model} (block {R.MTP_K + 1}, BF16 draft KV)")
    else:
        spec = f'{{"method":"mtp","num_speculative_tokens":{R.MTP_K},"draft_sample_method":"probabilistic"{draft_spec}}}'
    capture = _seqs_capture(srv["capture_sizes"], seqs, str(srv["max_num_seqs"]), 0 if spec is None else R.MTP_K)
    args = [
        "--served-model-name", *names,
        "--host", env.get("HOST") or "127.0.0.1", "--port", env.get("PORT") or str(base.get("port", 8000)),
        "--tensor-parallel-size", str(lay["tp"]), "--pipeline-parallel-size", str(lay["pp"]),
    ]
    if lay["ep"]:
        args.append("--enable-expert-parallel")
    if mode.get("disable_custom_all_reduce"):
        args.append("--disable-custom-all-reduce")
    if spec is not None:
        args += ["--speculative-config", spec]
    args += [
        "--max-model-len", maxlen, "--max-num-seqs", seqs,
        "--max-num-batched-tokens", chunk,
        "--gpu-memory-utilization", str(base["gpu_memory_utilization"]),
        "--kv-cache-memory-bytes", kv_bytes, "--kv-cache-dtype", srv["kv_cache_dtype"],
        "--limit-mm-per-prompt", '{"image":42,"video":0}', "--mm-processor-cache-gb", "0",
        "--mm-processor-kwargs", f'{{"size":{{"longest_edge":{img_cap_px},"shortest_edge":65536}}}}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[' + ",".join(map(str, capture)) + "]}",
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details",
    ]
    if lpt is not None:
        args += ["--long-prefill-token-threshold", lpt]
    args += [
        "--no-enable-flashinfer-autotune",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--override-generation-config", FN_GENERATION,
        *warm_args,
        *template_args,
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": True, "reasoning_effort": base.get("reasoning_effort", R.REASONING_EFFORT_DEFAULT)}),
        "--shutdown-timeout", "60",
        "--block-size", block, "--use-replayssm",
        "--prefix-match-unit", "64",
    ]
    # The GDN recurrent state dtype comes from the checkpoint's mamba_ssm_dtype alone; without it vLLM falls back to the
    # model dtype (bf16). Every served Flash-Next checkpoint declares float32: hold a new one to that.
    try:
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = None
    if isinstance(cfg, dict) and "mamba_ssm_dtype" not in cfg and "mamba_ssm_dtype" not in (cfg.get("text_config") or {}):
        args += ["--mamba-ssm-cache-dtype", "float32"]
        notes.append("note: the checkpoint's config.json has no mamba_ssm_dtype; GDN state runs float32 like every served "
                     "Flash-Next checkpoint (--mamba-ssm-cache-dtype float32), not the bf16 model-dtype fallback")
    plan = Plan("flash-next", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root,
                R.label_text(reg, p), p2p_mode, notes=notes, files=files,
                scales_recipe=scales_recipe)
    plan.topo = {"on": topo_on, "strict": auto_topo == "strict", "platform_env": platform_env, "here": here}
    plan.caller_env = dict(environ_in)
    plan.notes += [f"WARNING: {w}" for w in dflash2_override_warnings(p, environ_in)]
    return plan


def _build_27b(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    _spec_mtp_only(env, checkpoint or "27b")
    environ_in = dict(env)
    L = reg["launch"]
    shape = L["shapes"][p["launch_shape"]]
    ck = L["checkpoints"][checkpoint]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    here = f"{root}/models/qwen38-27b"
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", ck["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or shape["gpus"], True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    lpt = _knob(env, "LPT", str(shape["lpt"]), True)
    if not _digits(lpt):
        raise LaunchError(f"LPT must be a whole number of tokens (0 turns the cap off; got '{lpt}')", 2)
    if shape.get("spec_k_fixed"):
        spec_k = str(srv["spec_k"])
        if "SPEC_K" in env:
            env["SPEC_K"] = spec_k
    else:
        spec_k = _knob(env, "SPEC_K", str(srv["spec_k"]), True)
    max_pixels = _knob(env, "MAX_PIXELS", str(shape.get("max_pixels", L["max_pixels"])), True, keep_empty=True)
    if shape.get("vision_peak"):
        env["VLLM_MERLIN_VISION_PEAK"] = _knob(env, "VLLM_MERLIN_VISION_PEAK", "1", True, keep_empty=True)
    levers = shape.get("image_levers")
    if levers:  # one card: the image headroom settings (output-identical; each env variable overrides)
        env["VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"] = _knob(
            env, "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE", str(levers["flashinfer_workspace_bytes"]), True
        )
        env["MERLIN_SHARE_DRAFT_WORKSPACE"] = _knob(env, "MERLIN_SHARE_DRAFT_WORKSPACE", "1", True)
        env["VLLM_MERLIN_VISION_ROTARY_INPLACE"] = _knob(env, "VLLM_MERLIN_VISION_ROTARY_INPLACE", "1", True)
    chat_template = _knob(env, "CHAT_TEMPLATE", f"{here}/{L['chat_template']}", True, keep_empty=True)
    thinking = _knob(env, "THINKING", ck["thinking"], True)
    if ck.get("export_thinking"):  # the ThinkingCap wrappers export THINKING before forwarding
        env["THINKING"] = thinking
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    p2p_mode = None
    dropped_disable = None
    if shape.get("p2p_switch"):
        # a no-P2P row (27b-tp2/tp4-3090-nop2p) defaults to P2P off; the others default to on and take P2P=on|off
        p2p_mode = _knob(env, "P2P", "off" if p["match"].get("p2p") == "off" else "on", True)
        if p2p_mode == "on":
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["NCCL_P2P_LEVEL"] = "SYS"
            dropped_disable = env.pop("NCCL_P2P_DISABLE", None)  # an inherited =1 would contradict P2P=on
        elif p2p_mode == "off":
            env["VLLM_SKIP_P2P_CHECK"] = "1"
            env["NCCL_P2P_DISABLE"] = "1"
            for v in ("NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO"):
                env.pop(v, None)
        else:
            raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    smallq = _knob(env, "FLASHINFER_SM8X_FP8_SMALLQ", "cta32", True, keep_empty=True)
    env["FLASHINFER_SM8X_FP8_SMALLQ"] = smallq
    if smallq not in ("", "0", "cta32"):
        raise LaunchError(f"FLASHINFER_SM8X_FP8_SMALLQ must be empty, 0 or cta32 (got '{smallq}')", 2)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    env["HF_HUB_OFFLINE"] = _knob(env, "HF_HUB_OFFLINE", "1", True)

    def commit_env() -> None:
        env["VLLM_RECOVERSSM_COMMIT_GRAPH"] = _knob(env, "VLLM_RECOVERSSM_COMMIT_GRAPH", "graph", True, keep_empty=True)
        env["VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE"] = _knob(env, "VLLM_RECOVERSSM_COMMIT_AFTER_PROPOSE", "1", True, keep_empty=True)

    if shape["rssm_commit_env"] == "always":
        commit_env()
    rssm: list[str] = []
    if shape.get("recoverssm_fixed"):
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        rssm = ["--use-replayssm", "--mamba-backend", "triton"]
    else:
        default = "1" if srv["recoverssm"] else "0"
        rs = _knob(env, "RECOVERSSM", default, True)
        if rs == "1":
            env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            if shape["rssm_commit_env"] == "when_on":
                commit_env()
            rssm = ["--use-replayssm", "--mamba-backend", "triton"]
        elif rs != "0":
            raise LaunchError(f"RECOVERSSM must be {default} (default) or {'0' if default == '1' else '1'} (got '{rs}')", 2)
    # The host-RAM KV tier (merlin serve --kv-ram-cache): checked like Flash-Next's, and only on the RecoverSSM path
    # the engine qualifies it for (vllm/merlin/kv_host_tier.py).
    # ASCII digits only, and int() guarded: superscript two (U+00B2) passes isdigit() but int() refuses it, and so
    # does a value over Python's 4300-digit limit; either is a LaunchError, never a traceback (as on Flash-Next).
    tier = (environ_in.get(FN_KV_HOST_TIER) or "").strip()
    tier_gib = 0
    if tier:
        try:
            if not (tier.isascii() and tier.isdigit()):
                raise ValueError
            tier_gib = int(tier)
        except ValueError:
            shown = tier if len(tier) <= 40 else f"{tier[:20]}...({len(tier)} characters)"
            raise LaunchError(f"{FN_KV_HOST_TIER} must be a whole number of GiB per rank (0 = off), got {shown!r}",
                              2) from None
    if tier_gib > 0:
        if not rssm:
            raise LaunchError(f"{FN_KV_HOST_TIER}={tier_gib}: the KV RAM cache is qualified on the 27B's RecoverSSM path "
                              "only; drop RECOVERSSM=0 or set the cache to 0 (off)", 2)
        _restore_unit_gate("27b", p, tier_gib)
        env[FN_KV_HOST_TIER] = str(tier_gib)
    else:
        env.pop(FN_KV_HOST_TIER, None)
    # The pacing knobs: checked as on Flash-Next (the 27B passes its environment through, so a blank export reached the
    # engine's strict parser and failed the boot).
    _kv_tier_gates(environ_in, env)
    # The debug census: checked as on Flash-Next.
    census_note = _kv_tier_census(environ_in, env)
    extra_flags: list[str] = []
    if chat_template:
        extra_flags += ["--chat-template", chat_template]
    _max_pixels_27b(max_pixels)
    if max_pixels:
        extra_flags += ["--mm-processor-kwargs", f'{{"size":{{"longest_edge":{max_pixels},"shortest_edge":65536}}}}']
    if spec_k not in ("1", "3", "4"):
        raise LaunchError("SPEC_K must be 1, 3 or 4 (the measured values; 4 is the default)", 2)
    if shape.get("spec_k_fixed"):
        capture = srv["capture_sizes"]
    else:
        capture = srv["capture_sizes"] if spec_k == "4" else shape["capture_sizes_k_other"]
    lpt_flags = [] if lpt == "0" else ["--long-prefill-token-threshold", lpt]
    kvf = [] if kv_bytes == "auto" else ["--kv-cache-memory-bytes", kv_bytes]
    chunk = _int_knob(env, "CHUNK", True, str(srv["max_num_batched_tokens"]), 2)
    block = _int_knob(env, "BLOCK_SIZE", True, str(srv["block_size"]) if "block_size" in srv else None, 2)
    notes: list[str] = []
    seqs, maxlen = _shape_knobs(env, _session_cap(p, environ_in, notes), str(L["max_model_len"]), 2, notes)
    capture = _seqs_capture(capture, seqs, str(srv["max_num_seqs"]), int(spec_k))
    args = [
        "--served-model-name", served,
        "--tensor-parallel-size", str(srv["tensor_parallel"]), "--attention-backend", "FLASHINFER",
        "--max-model-len", maxlen, "--gpu-memory-utilization", "0.96",
        "--kv-cache-dtype", srv.get("kv_cache_dtype", "fp8_e4m3"), *kvf, "--mamba-ssm-cache-dtype", "float16",
        "--max-num-seqs", seqs, "--max-num-batched-tokens", chunk,
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details", *lpt_flags,
        "--prefix-match-unit", "16", "--enable-mamba-shared-prefix-checkpoint", *rssm,
    ]
    if block is not None:
        args += ["--block-size", block]
    if shape.get("disable_custom_all_reduce"):
        args.append("--disable-custom-all-reduce")
    args += [
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{spec_k},"draft_sample_method":"probabilistic"}}',
        "--compilation-config",
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[' + ",".join(map(str, capture))
        + '],"inductor_compile_config":' + B27_INDUCTOR + "}",
        "--override-generation-config", B27_GENERATION,
        "--default-chat-template-kwargs", thinking,
        "--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
    ]
    if shape.get("cpu_offload_visual_gb"):
        args += ["--cpu-offload-gb", str(shape["cpu_offload_visual_gb"]), "--cpu-offload-params", "visual"]
    args += [
        "--limit-mm-per-prompt", f'{{"image":{shape["images_per_prompt"]},"video":0}}', "--mm-processor-cache-gb", "0",
        *extra_flags,
        "--host", host, "--port", port, "--shutdown-timeout", "60",
    ]
    # Keep this install's caches separate from a user's old global cache; explicit vLLM / Triton overrides still win.
    cache_root = cache_root_for("27b", env, repo)
    env["VLLM_CACHE_ROOT"] = cache_root
    if not env.get("TRITON_CACHE_DIR"):
        env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    plan = Plan("27b", checkpoint, p, ckpt, env, args, extra, repo, venv, None, cache_root,
                R.label_text(reg, p), p2p_mode, notes=notes)
    plan.caller_env = environ_in
    if env.get(FN_KV_HOST_TIER):
        plan.notes.append(KV_TIER_EXPERIMENTAL_27B)
    if census_note:
        plan.notes.append(census_note)
    if dropped_disable is not None:
        plan.notes.append(f"note: NCCL_P2P_DISABLE={dropped_disable!r} dropped: P2P=on; on a host without peer-to-peer "
                          "set P2P=off (merlin serve picks the no-P2P row)")
    return plan


def _build_svl(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    _spec_mtp_only(env, "nemotron-super-vl")
    environ_in = dict(env)
    L = reg["launch"]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or p.get("cuda_visible_devices", "0,1,2,3"), True)
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", L["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    cache_root = _knob(env, "CACHE_ROOT", f"{root}/.vllm-cache", True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    mnbt = _knob(env, "MNBT", str(srv["max_num_batched_tokens"]), True)
    chunk = _int_knob(env, "CHUNK", True, mnbt, 2)
    seqs = _knob(env, "SEQS", str(srv["max_num_seqs"]), True)
    maxlen = _knob(env, "MAXLEN", str(L["max_model_len"]), True)
    blk = _knob(env, "BLOCK", str(srv["block_size"]), True)
    block = _int_knob(env, "BLOCK_SIZE", True, blk, 2)
    lpt = _int_knob(env, "LPT", True, None, 2)
    partition = _knob(env, "PP_PARTITION", ",".join(map(str, p["layout"]["partition"])), True, keep_empty=True)
    max_pixels = _knob(env, "MAX_PIXELS", "", True, keep_empty=True)
    if max_pixels != "":
        if max_pixels.startswith("0") or not _digits(max_pixels):
            raise LaunchError("MAX_PIXELS must be empty or a canonical decimal pixel count (digits only, no leading zero)", 2)
        if len(max_pixels) > 10 or int(max_pixels) < 8388608:
            raise LaunchError(f"MAX_PIXELS must be >= 8388608 (the 8 MP floor) (got {max_pixels})", 2)
    if not _digits(kv_bytes) or int(kv_bytes) < 1073741824:
        raise LaunchError(f"KV_BYTES must be integer bytes >= 1 GiB (never a util fraction): {kv_bytes}", 2)
    if not re.fullmatch(r"|[0-9][^,]*,[0-9].*", partition):
        raise LaunchError(f"PP_PARTITION must be empty or 'n,m' layer counts (got '{partition}')", 2)
    # The host-RAM KV tier is not qualified on Super VL, which passes its environment through: an
    # active tier, census or pacing setting is refused before the boot; inactive ones (0 or blank) are not exported.
    tier = (environ_in.get(FN_KV_HOST_TIER) or "").strip()
    if tier and not (tier.isascii() and tier.isdigit() and not tier.strip("0")):
        shown = tier if len(tier) <= 40 else f"{tier[:20]}...({len(tier)} characters)"
        raise LaunchError(f"{FN_KV_HOST_TIER}={shown!r}: the KV RAM cache is not qualified on Super VL; set it to 0 "
                          "(off) or leave it unset", 2)
    env.pop(FN_KV_HOST_TIER, None)
    _kv_tier_gates(environ_in, env)
    _kv_tier_census(environ_in, env)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # a no-P2P row (svl-quad-3090-nop2p) defaults to P2P off; the others default to on and take P2P=on|off
    p2p_mode = _knob(env, "P2P", "off" if p["match"].get("p2p") == "off" else "on", True)
    p2p_args: list[str] = []
    if p2p_mode == "on":
        # Super VL keeps custom all-reduce with NCCL LL: the hybrid all-reduce changed its temperature-0 output.
        env["NCCL_P2P_LEVEL"], env["NCCL_PROTO"], env["VLLM_SKIP_P2P_CHECK"] = "SYS", "LL", "1"
        env.pop("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC", None)
    elif p2p_mode == "off":
        env["VLLM_SKIP_P2P_CHECK"], env["NCCL_P2P_DISABLE"] = "1", "1"
        for v in ("NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"):
            env.pop(v, None)
        p2p_args = ["--disable-custom-all-reduce"]
    else:
        raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    env["VLLM_CACHE_ROOT"] = cache_root
    env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    master, dist = distributed_ports(port)
    env["MASTER_PORT"], env["VLLM_PORT"] = str(master), str(dist)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    env["HF_HUB_OFFLINE"] = _knob(env, "HF_HUB_OFFLINE", "1", True)
    env["TRANSFORMERS_OFFLINE"] = _knob(env, "TRANSFORMERS_OFFLINE", "1", True)
    if partition:
        env["VLLM_PP_LAYER_PARTITION"] = partition
    if max_pixels:
        env["VLLM_MAX_IMAGE_PIXELS"] = max_pixels
    tree = env.get("TREE") or None
    if tree:
        env["PYTHONPATH"] = tree + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    lpt_flags = [] if lpt is None else ["--long-prefill-token-threshold", lpt]
    args = [
        "--served-model-name", served, "--trust-remote-code",
        "--tensor-parallel-size", str(p["layout"]["tp"]), "--pipeline-parallel-size", str(p["layout"]["pp"]),
        "--enable-expert-parallel",
        "--max-model-len", maxlen, "--max-num-seqs", seqs, "--max-num-batched-tokens", chunk, *lpt_flags,
        "--kv-cache-dtype", L["kv_cache_dtype_flag"] if srv.get("kv_cache_dtype", "fp8_e4m3") == "fp8_e4m3" else srv["kv_cache_dtype"],
        "--kv-cache-memory-bytes", kv_bytes, "--gpu-memory-utilization", "0.96",
        "--mamba-ssm-cache-dtype", "float32", "--attention-backend", "FLASHINFER",
        "--block-size", block, "--enable-prefix-caching", "--prefix-match-unit", blk, "--mamba-cache-mode", "align",
        "--use-replayssm", "--mamba-backend", "triton",
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{srv["spec_k"]}}}',
        "--reasoning-parser", "nemotron_v3", "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",  # the checkpoint card's vLLM recipe
        "--limit-mm-per-prompt", '{"image":1,"video":0}', "--mm-processor-cache-gb", "0",
        *p2p_args, "--host", host, "--port", port,
    ]
    plan = Plan("svl", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root, R.label_text(reg, p), p2p_mode)
    plan.caller_env = environ_in
    return plan


def _build_ornith(reg, p, checkpoint, ckpt, env, extra, repo: Path) -> Plan:
    """Ornith 1.5 (Qwen3.5-122B-A10B MoE fine-tune): TP2 x PP2 + EP, FP8 KV, MTP with RecoverSSM on Model Runner V2, the
    checkpoint's own chat template and sampling. Needs the engine's Qwen3.5-MoE RecoverSSM / host-embed / uniform-verify
    support in this engine."""
    _spec_mtp_only(env, "ornith1.5")
    environ_in = dict(env)
    notes: list[str] = []
    L = reg["launch"]
    srv = p["serve"]
    cwd = os.getcwd()
    root = repo.as_posix()
    gpus = _knob(env, "GPUS", env.get("CUDA_VISIBLE_DEVICES") or p.get("cuda_visible_devices", "0,1,2,3"), True)
    port = _knob(env, "PORT", str(L["port"]), True)
    host = _knob(env, "HOST", "127.0.0.1", True)
    served = _knob(env, "SERVED", L["served"], True)
    venv = _abs(_knob(env, "VENV", f"{root}/venv", True), cwd)
    if "VENV" in env:
        env["VENV"] = venv
    cache_root = _knob(env, "CACHE_ROOT", f"{root}/.vllm-cache", True)
    kv_bytes = _knob(env, "KV_BYTES", str(srv["kv_cache_memory"]), True)
    chunk = _int_knob(env, "CHUNK", True, str(srv["max_num_batched_tokens"]), 2)
    seqs, maxlen = _shape_knobs(env, str(srv["max_num_seqs"]), str(L["max_model_len"]), 2, notes)
    capture = _seqs_capture(L["capture_sizes"], seqs, str(srv["max_num_seqs"]), int(srv["spec_k"]))
    block = _int_knob(env, "BLOCK_SIZE", True, str(srv["block_size"]), 2)
    lpt = _int_knob(env, "LPT", True, None, 2)
    partition = _knob(env, "PP_PARTITION", ",".join(map(str, p["layout"]["partition"])), True, keep_empty=True)
    max_pixels = _knob(env, "MAX_PIXELS", str(srv["image_max_pixels"]), True)
    uv = _knob(env, "UV", "1" if srv.get("uniform_verify") else "0", True)
    if not _digits(max_pixels) or max_pixels.startswith("0") or len(max_pixels) > 10 or int(max_pixels) < 8388608:
        raise LaunchError(f"MAX_PIXELS must be a pixel count >= 8388608 (the 8 MP floor) (got {max_pixels})", 2)
    if not _digits(kv_bytes) or int(kv_bytes) < 1073741824:
        raise LaunchError(f"KV_BYTES must be integer bytes >= 1 GiB (never a util fraction): {kv_bytes}", 2)
    if block is None or int(block) < 4608:
        raise LaunchError(f"BLOCK_SIZE must be >= 4608 (the engine needs >= 4512 at fp32 SSM state) (got {block})", 2)
    if partition and not (re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*", partition) and sum(map(int, partition.split(","))) == 48):
        raise LaunchError(f"PP_PARTITION must be empty or two layer counts that sum to 48, e.g. 25,23 (got '{partition}')", 2)
    if uv not in ("0", "1"):
        raise LaunchError(f"UV must be 1 or 0 (got '{uv}')", 2)
    exported_uv = environ_in.get("MERLIN_MOE_UNIFORM_VERIFY")
    if exported_uv and exported_uv != uv:  # the engine name is overwritten below; UV is the launcher's knob
        notes.append(f"warning: MERLIN_MOE_UNIFORM_VERIFY={exported_uv} is ignored: the launcher sets it from UV={uv} "
                     "(the profile default unless UV is set); set UV instead")
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    row_p2p = "off" if p["match"].get("p2p") == "off" else "on"
    p2p_mode = _knob(env, "P2P", row_p2p, True)
    if p2p_mode in ("on", "off") and p2p_mode != row_p2p:
        other = "ornith-quad-3090" if p2p_mode == "on" else "ornith-quad-3090-nop2p"
        raise LaunchError(f"P2P={p2p_mode} does not match {p['id']} (qualified with P2P {row_p2p}; its KV pin was measured on that "
                          f"interconnect): use --profile {other}", 2)
    if p2p_mode == "on":
        env["VLLM_SKIP_P2P_CHECK"], env["NCCL_P2P_LEVEL"], env["NCCL_PROTO"] = "1", "SYS", "Simple"
        env["VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"] = "1"
        env.pop("NCCL_P2P_DISABLE", None)
    elif p2p_mode == "off":
        env["NCCL_P2P_DISABLE"] = "1"
        for v in ("VLLM_SKIP_P2P_CHECK", "NCCL_P2P_LEVEL", "NCCL_PROTO", "NCCL_ALGO", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC"):
            env.pop(v, None)
    else:
        raise LaunchError(f"P2P must be on or off (got '{p2p_mode}')", 2)
    # The host-RAM KV tier (merlin serve --kv-ram-cache), an untested opt-in on Ornith: parsed as on the 27B, and held to
    # the restore-unit minimum, which falls back to 4 GiB per server because no Ornith row records its KV block count.
    # Ornith always runs RecoverSSM on Model Runner V2 in align mode, the path the engine accepts it on
    # (vllm/merlin/kv_host_tier.py).
    # ASCII digits only, and int() guarded: superscript two (U+00B2) passes isdigit() but int() refuses it, and so
    # does a value over Python's 4300-digit limit; either is a LaunchError, never a traceback (as on Flash-Next).
    tier = (environ_in.get(FN_KV_HOST_TIER) or "").strip()
    tier_gib = 0
    if tier:
        try:
            if not (tier.isascii() and tier.isdigit()):
                raise ValueError
            tier_gib = int(tier)
        except ValueError:
            shown = tier if len(tier) <= 40 else f"{tier[:20]}...({len(tier)} characters)"
            raise LaunchError(f"{FN_KV_HOST_TIER} must be a whole number of GiB per rank (0 = off), got {shown!r}",
                              2) from None
    if tier_gib > 0:
        _restore_unit_gate("ornith", p, tier_gib)
        env[FN_KV_HOST_TIER] = str(tier_gib)
    else:
        env.pop(FN_KV_HOST_TIER, None)
    # The pacing knobs and the debug census: checked as on the other families (Ornith passes its environment through
    # like the 27B).
    _kv_tier_gates(environ_in, env)
    census_note = _kv_tier_census(environ_in, env)
    if census_note:
        notes.append(census_note)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"
    env["MERLIN_MOE_UNIFORM_VERIFY"] = uv
    env["VLLM_CACHE_ROOT"] = cache_root
    env["TRITON_CACHE_DIR"] = f"{cache_root}/triton"
    master, dist = distributed_ports(port)
    env["MASTER_PORT"], env["VLLM_PORT"] = str(master), str(dist)
    env["VLLM_NO_USAGE_STATS"] = _knob(env, "VLLM_NO_USAGE_STATS", "1", True)
    env["DO_NOT_TRACK"] = _knob(env, "DO_NOT_TRACK", "1", True)
    env["HF_HUB_OFFLINE"] = _knob(env, "HF_HUB_OFFLINE", "1", True)
    if partition:
        env["VLLM_PP_LAYER_PARTITION"] = partition
    else:
        env.pop("VLLM_PP_LAYER_PARTITION", None)  # PP_PARTITION="" = the engine's even split, never an inherited value
    tree = env.get("TREE") or None
    if tree:  # launcher convention (as for Super VL): the tree goes first, an inherited PYTHONPATH stays behind it
        env["PYTHONPATH"] = tree + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    lpt_flags = [] if lpt is None else ["--long-prefill-token-threshold", lpt]
    args = [
        "--served-model-name", served,
        "--tensor-parallel-size", str(p["layout"]["tp"]), "--pipeline-parallel-size", str(p["layout"]["pp"]),
        "--disable-custom-all-reduce",
        "--max-model-len", maxlen, "--max-num-seqs", seqs, "--max-num-batched-tokens", chunk, *lpt_flags,
        "--gpu-memory-utilization", "0.96", "--kv-cache-memory-bytes", kv_bytes,
        "--kv-cache-dtype", srv.get("kv_cache_dtype", "fp8_e4m3"), "--block-size", block,
        "--limit-mm-per-prompt", f'{{"image":{L["images_per_prompt"]},"video":0}}', "--mm-processor-cache-gb", "0",
        "--mm-processor-kwargs", f'{{"size":{{"longest_edge":{max_pixels},"shortest_edge":65536}}}}',
        "--enable-prefix-caching", "--mamba-cache-mode", "align", "--prefix-match-unit", str(srv["prefix_match_unit"]),
        "--enable-prompt-tokens-details", "--no-enable-flashinfer-autotune",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
        "--override-generation-config", ORN_GENERATION, "--kernel-config", ORN_KERNEL_CONFIG,
        "--mamba-backend", "triton", "--shutdown-timeout", "60",
    ]
    if p["layout"].get("ep"):
        args.append("--enable-expert-parallel")
    args += [
        "--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{srv["spec_k"]},"draft_sample_method":"probabilistic"}}',
        "--use-replayssm",
        "--compilation-config", '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":['
        + ",".join(map(str, capture)) + "]}",
        "--host", host, "--port", port,
    ]
    plan = Plan("ornith", checkpoint, p, ckpt, env, args, extra, repo, venv, tree, cache_root, R.label_text(reg, p), p2p_mode,
                notes=notes)
    plan.caller_env = environ_in
    return plan


# --------------------------------------------------------------------------------------------
# Host phase: checks, toolchain, topology, then exec
# --------------------------------------------------------------------------------------------
def _say(msg: str) -> None:
    print(msg, flush=True)


def _venv_python(venv: str, code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([f"{venv}/bin/python", "-c", code, *args], capture_output=True, text=True)


def _verify_products(tree: str) -> None:
    sums = f"{tree}/.build-products.sha256"
    if not os.path.isfile(f"{tree}/.build-products") or not (os.path.isfile(sums) and os.path.getsize(sums) > 0):
        raise LaunchError(f"TREE={tree} has no verified build products (release/install-env.sh)", 1)
    ok = True
    with open(f"{tree}/.build-products.sha256", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    for ln in lines:  # sha256sum -c --strict: every line must parse and match
        m = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", ln)
        if not m:
            ok = False
            break
        path = os.path.join(tree, m.group(2))
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            ok = False
            break
        if digest.lower() != m.group(1).lower():
            ok = False
            break
    if not ok:
        raise LaunchError(f"TREE={tree}: build products changed since release/install-env.sh verified them", 1)


def _check_shadow(tree: str) -> None:
    pwd_vllm = os.path.join(os.getcwd(), "vllm")
    if not os.path.isdir(pwd_vllm):
        return
    real = os.path.realpath(pwd_vllm)
    if real == os.path.realpath(os.path.join(tree, "vllm")):
        return
    if real == os.path.realpath(tree) and not os.path.exists(os.path.join(tree, "__init__.py")):
        return
    raise LaunchError(
        f"refusing to start: {pwd_vllm} would shadow {tree}/vllm.\n"
        "  The registry-inspection child puts the current directory before PYTHONPATH on sys.path, so that copy\n"
        "  would be imported instead of the built tree. Start the server from a directory without a vllm/ beside it,\n"
        "  or pass TREE=<that checkout> if it really is the built tree.", 1)


def nvcc_release(cuda_home: str) -> str | None:
    try:
        out = subprocess.run([f"{cuda_home}/bin/nvcc", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return None
    m = re.search(r"release (\d+\.\d+),", out)
    return m.group(1) if m else None


def check_inherited_cuda_home(cuda_home: str) -> None:
    """A CUDA_HOME from the caller's shell replaces the venv's locked CUDA 13.0 toolkit; JIT kernels built with
    another major version do not load against torch's CUDA 13 build."""
    rel = nvcc_release(cuda_home)
    if rel is None:
        _say(f"merlin: warning: CUDA_HOME={cuda_home} (from the environment) has no working nvcc; "
             "unset CUDA_HOME to use the venv's locked toolkit")
    elif rel.split(".")[0] != "13":
        raise LaunchError(
            f"CUDA_HOME={cuda_home} (from the environment) is CUDA {rel}; this engine needs a CUDA 13 toolkit. "
            "Unset CUDA_HOME to use the venv's locked toolkit.", 1)


def _toolchain(plan: Plan, full: bool) -> None:
    """CUDA_HOME default = the venv's locked CUDA 13.0 toolkit (release/install-0.32.md); ``full`` adds the
    first-serve checks of the Flash-Next recipe (CC or cc, CXX or c++, ninja, Python.h, nvcc, libcudart, nvcc-vs-driver).
    The venv's bin follows the CUDA tools on PATH: the lock installs Ninja there."""
    env, venv = plan.env, plan.venv
    if not env.get("CUDA_HOME"):
        cp = _venv_python(venv, 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
        if cp.returncode != 0:
            raise LaunchError(f"VENV={venv}: cannot read the venv's purelib for the CUDA_HOME default ({cp.stderr.strip()})", 1)
        env["CUDA_HOME"] = cp.stdout.strip().splitlines()[-1] + "/nvidia/cu13"
    else:
        check_inherited_cuda_home(env["CUDA_HOME"])
    cuda_home = env["CUDA_HOME"]
    env["PATH"] = f"{cuda_home}/bin:{venv}/bin:" + env.get("PATH", "")
    if not full:
        return
    import shutil

    for t in (env.get("CC") or "cc", env.get("CXX") or "c++", "ninja"):
        if shutil.which(t, path=env["PATH"]) is None:
            raise LaunchError(f"first serve needs {t} (Debian/Ubuntu: apt install gcc g++ ninja-build)", 1)
    cp = _venv_python(
        venv,
        'import os,sys,sysconfig; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)',
    )
    if cp.returncode != 0:
        raise LaunchError(
            "first serve needs the Python 3.13 headers: Triton compiles a C module against Python.h (Debian 13, or Ubuntu "
            "with the deadsnakes PPA: apt install python3.13-dev; uv and pyenv Pythons ship Python.h; the venv must come "
            "from that interpreter)", 1)
    nvcc = f"{cuda_home}/bin/nvcc"
    if not os.access(nvcc, os.X_OK):
        raise LaunchError(f"no nvcc at CUDA_HOME={cuda_home} (release/install-env.sh installs the CUDA 13.0 compiler wheels)", 1)
    if not os.path.exists(f"{cuda_home}/lib64/libcudart.so"):
        raise LaunchError(f"{cuda_home}/lib64/libcudart.so missing (release/install-env.sh creates the lib64 links)", 1)
    nv_rel = nvcc_release(cuda_home)
    if not nv_rel:
        raise LaunchError(f"could not read a CUDA release from `{nvcc} --version` (see its error above)", 1)
    min_drv = "13.0" if nv_rel.split(".")[0] == "13" else nv_rel
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, env=env).stdout
    except OSError:
        smi = ""
    dm = re.search(r"CUDA (?:UMD )?Version: *(\d+\.\d+)", smi)
    if not dm:
        _say("note: could not read the driver's CUDA version from nvidia-smi; the nvcc-versus-driver check is skipped")
        return
    drv = dm.group(1)
    if tuple(map(int, drv.split("."))) < tuple(map(int, min_drv.split("."))):
        raise LaunchError(
            f"nvcc at {cuda_home} is CUDA {nv_rel} but the driver supports CUDA {drv}; this toolkit requires CUDA {min_drv} or later.\n"
            "  Upgrade the driver or set CUDA_HOME to a compatible toolkit (see release/install-0.32.md).", 1)


_TOPO_VAL = r"([A-Za-z0-9_@%+=:,./-]+|'[^']*'(\"'\"'[^']*')*)"
_TOPO_LINES = [
    r"# topo_select v1 status=(ok|legacy) reason=[A-Za-z0-9_.-]+",
    rf"export CUDA_VISIBLE_DEVICES={_TOPO_VAL}",
    rf"export CUDA_DEVICE_ORDER={_TOPO_VAL}",
    rf"export VLLM_SKIP_P2P_CHECK={_TOPO_VAL}",
    rf"(export NCCL_P2P_LEVEL={_TOPO_VAL}|unset NCCL_P2P_LEVEL)",
    rf"(export NCCL_PROTO={_TOPO_VAL}|unset NCCL_PROTO)",
    r"export TOPO_LAUNCH_PREFIX=('numactl --physcpubind=[0-9,-]+ --preferred=[0-9]+ --'|'')",
    rf"export TOPO_DECISION_JSON={_TOPO_VAL}",
]


def _topo_env_file(path: str) -> list[str] | None:
    """The selector's env file must be exactly its 8-line format; anything else means 'do not source it'."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError:
        return None
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != len(_TOPO_LINES):
        return None
    for ln, pat in zip(lines, _TOPO_LINES):
        if not re.fullmatch(pat, ln):
            return None
    return lines


def _apply_topo_lines(env: dict[str, str], lines: list[str]) -> str:
    prefix = ""
    for ln in lines[1:]:
        if ln.startswith("unset "):
            env.pop(ln[6:], None)
            continue
        name, val = ln[7:].split("=", 1)
        val = shlex.split(val)[0] if val.startswith("'") else val
        env[name] = val
        if name == "TOPO_LAUNCH_PREFIX":
            prefix = val
    return prefix


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, AttributeError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def _topology(plan: Plan) -> None:
    """The AUTO_TOPO selector integration (models/flash-next/docs/topology-selector.md)."""
    t = plan.topo
    env = plan.env
    plan.prefix = []
    if not t["on"]:
        return
    snapshot: dict[str, str | None] = {}
    for v in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_SKIP_P2P_CHECK", "NCCL_P2P_LEVEL", "NCCL_PROTO", "PLE_HOME"):
        caller = plan.caller_env.get(v)
        snapshot[v] = caller
        if caller is not None:
            env[f"TOPO_CALLER_SET_{v}"], env[f"TOPO_CALLER_{v}"] = "1", caller
        else:
            env[f"TOPO_CALLER_SET_{v}"] = "0"
    env["TOPO_LAUNCH_PREFIX"] = ""
    env["TOPO_DECISION_JSON"] = ""
    timeout = 20
    raw = env.get("TOPO_TIMEOUT", "")
    if raw != "":
        if re.fullmatch(r"[1-9][0-9]{0,2}", raw):
            if int(raw) <= 600:
                timeout = int(raw)
            else:
                _say(f"topo: note: TOPO_TIMEOUT={raw} is above 600 s; using 20")
        else:
            _say(f"topo: note: TOPO_TIMEOUT='{raw}' is not a whole number of seconds from 1 to 600; using 20")

    def legacy(fallback: bool) -> None:
        if fallback:
            cvd = snapshot["CUDA_VISIBLE_DEVICES"]
            if cvd is None or cvd.strip() == "":
                env.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                env["CUDA_VISIBLE_DEVICES"] = cvd
            env["TOPO_LAUNCH_PREFIX"] = ""
            env["TOPO_DECISION_JSON"] = ""
            plan.prefix = []
        t["platform_env"](env)

    def unavailable(why: str) -> None:
        if t["strict"]:
            _say(f"topo: {why}; AUTO_TOPO=strict refuses to start without a selector decision (AUTO_TOPO=1 falls back to the legacy settings)")
            raise LaunchError("", 1)
        _say(f"topo: {why}, using legacy settings")
        legacy(True)

    # the platform block ran in build() with the caller's CUDA_VISIBLE_DEVICES; the selector decides from the snapshot
    try:
        # like mktemp(1): honour TMPDIR as given (a missing TMPDIR is a failure, not a fallback)
        fd, tmp = tempfile.mkstemp(dir=os.environ.get("TMPDIR") or None)
        os.close(fd)
    except OSError:
        unavailable("could not create a temporary file for the selector")
        return
    try:
        sel_env = dict(env)
        sel_env["TOPO_TIMEOUT"] = str(timeout)
        for v, val in snapshot.items():  # the selector sees the caller's values, not our defaults
            if val is None:
                sel_env.pop(v, None)
            else:
                sel_env[v] = val
        cmd = [f"{plan.venv}/bin/python", f"{t['here']}/topo_select.py", "--env-out", tmp, "--json-dir", f"{plan.cache_root}/topology"]
        rc = 0
        try:
            # its own process group, so the watchdog ends the selector AND anything it spawned (as `timeout` does)
            proc = subprocess.Popen(cmd, env=sel_env, start_new_session=True)
            try:
                rc = proc.wait(timeout=timeout + 5)
            except subprocess.TimeoutExpired:
                _signal_group(proc, 15)
                try:
                    proc.wait(timeout=5)
                    rc = 124
                except subprocess.TimeoutExpired:
                    _signal_group(proc, 9)
                    proc.wait()
                    rc = 137
        except OSError:
            rc = 127
        if rc == 3:
            raise LaunchError("", 1)  # strict mode refused to guess; the selector already put its reason on stderr
        if rc != 0:
            unavailable(f"selector unavailable (rc={rc})")
            return
        lines = _topo_env_file(tmp)
        if lines is None:
            unavailable("selector unavailable (rc=0, unusable env file)")
            return
        plan.prefix = shlex.split(_apply_topo_lines(env, lines))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _cgroup_bytes(path: str) -> int | None:
    """A cgroup memory file as an integer; None when missing, ``max``, or a limit so large it means unlimited."""
    v = (_read_text(path) or "").strip()
    n = _small_whole(v)
    if n is None or n >= 1 << 60:
        return None
    return n


# (mount, limit, usage, memory.stat key of the reclaimable page cache) for cgroup v2 and v1; preflight.CGROUP_MEMORY_SETS
_CGROUP_MEMORY_SETS = (
    ("/sys/fs/cgroup", "memory.max", "memory.current", "inactive_file"),
    ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"),
)


def _cgroup_memory_dirs(mount: str, limit_file: str) -> list[str]:
    """This process's memory cgroup and every level above it under ``mount``, the mount last: the same rule as
    preflight.cgroup_memory_dirs (from /proc/self/cgroup, so a systemd MemoryMax on a unit or scope is seen; just the
    mount when that file is unreadable or names the root or a path that is not a cgroup under the mount)."""
    v2 = mount == "/sys/fs/cgroup"
    rel = None
    for ln in (_read_text("/proc/self/cgroup") or "").splitlines():
        parts = ln.strip().split(":", 2)
        if len(parts) != 3:
            continue
        if (v2 and parts[0] == "0" and parts[1] == "") or (not v2 and "memory" in parts[1].split(",")):
            rel = parts[2]
            break
    segs = [x for x in (rel or "").split("/") if x and x not in (".", "..")]
    dirs = [mount + "/" + "/".join(segs[:i]) for i in range(len(segs), 0, -1)]
    if not dirs or (_read_text(f"{dirs[0]}/{limit_file}") is None and _read_text(f"{dirs[0]}/cgroup.procs") is None):
        return [mount]
    return dirs + [mount]


def _cgroup_stat_bytes(path: str, key: str) -> int | None:
    """One ``key value`` line of a cgroup memory.stat; None when the file or the key is missing."""
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(rf"(?m)^{re.escape(key)}\s+(\d+)\s*$", f.read())
    except OSError:
        return None
    return int(m.group(1)) if m else None


def _host_memory_warnings(workers: int = 4, tier_gib: int = 0, ple_gib: float = FN_PLE_FP8_GIB) -> str | None:
    # Warns on the live memory reading, as the hand-written launchers did; host_phase refuses on the one reading that
    # cannot recover, a host RAM (MemTotal or the cgroup limit) under the resident floor, with the returned text.
    # merlin serve refuses on the same reading (serve_cli.py). ``workers`` = TP x PP of the layout: the resident floor
    # is the measured four-worker figure plus ~2 GiB per worker above four, plus the KV host tier's arena of
    # ``tier_gib`` per rank when it is on (preflight.ram_floor_gib; the qualified 96 GB is for the tier off), with the
    # FP8 table term swapped for ``ple_gib`` (the checkpoint's table: INT6 g32 is 38.74 GiB).
    tier_kb = max(0, tier_gib) * workers * 1048576
    floor_kb = FN_RAM_FLOOR_KB + max(0, workers - 4) * 2097152 + tier_kb - int(round((FN_PLE_FP8_GIB - ple_gib) * 1048576))
    floor_gib = floor_kb // 1048576
    tier_s = (f" + a KV host tier of {tier_gib} GiB x {workers} ranks = {tier_kb // 1048576} GiB" if tier_kb else "")
    mem_kb: int | None = None
    avail_kb: int | None = None
    refusal: str | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("MemTotal:"):
                    mem_kb = int(ln.split()[1])
                elif ln.startswith("MemAvailable:"):
                    avail_kb = int(ln.split()[1])
    except (OSError, ValueError):
        pass
    # MemAvailable capped by what every cgroup level from this process's own up to the root has left:
    # max - (current - inactive_file), the reclaimable page cache taken out of the usage; MemTotal lowered to the
    # smallest limit on the way. The same rule as preflight.mem_available_mib / host_ram_mib (kept inline: the
    # launcher runs from trees without preflight.py)
    for mount, lim_name, use_name, key in _CGROUP_MEMORY_SETS:
        for d in _cgroup_memory_dirs(mount, lim_name):
            lim, use = _cgroup_bytes(f"{d}/{lim_name}"), _cgroup_bytes(f"{d}/{use_name}")
            if lim is not None:
                lim_kb = lim // 1024
                if mem_kb is None or lim_kb < mem_kb:
                    mem_kb = lim_kb
            if lim is not None and use is not None:
                use = max(0, use - (_cgroup_stat_bytes(f"{d}/memory.stat", key) or 0))
                left_kb = max(0, lim - use) // 1024
                if avail_kb is None or left_kb < avail_kb:
                    avail_kb = left_kb
    if mem_kb is not None and (mem_kb < FN_RAM_QUALIFIED_KB + tier_kb or mem_kb < floor_kb):
        _say(f"WARNING: host RAM (MemTotal or the cgroup limit) is {mem_kb // 1048576} GiB; the qualified configuration is 96 GB"
             + (f" plus the KV host tier ({tier_kb // 1048576} GiB)" if tier_kb else ""))
        _say(f"         (measured resident floor ~{floor_gib} GiB for {workers} workers: ~{ple_gib:.0f} GiB n-gram table + ~4.2 GiB pinned "
             f"embeddings + workers{tier_s}).")
        if mem_kb < floor_kb:
            _say(f"         Below the ~{floor_gib} GiB resident floor the model load is expected to OOM.")
            refusal = (f"host RAM (MemTotal or the cgroup limit) is {mem_kb // 1048576} GiB, below the ~{floor_gib} GiB resident "
                       f"floor for {workers} workers{tier_s}: the model load would be OOM-killed minutes into the boot "
                       f"(four workers, tier off: 64 GiB was OOM-killed, {FN_RAM_TESTED_MIN_GIB} GiB is the tested minimum). "
                       "Raise the memory limit or add RAM"
                       + (", or lower --kv-ram-cache" if tier_kb else "") + ".")
    if avail_kb is not None and avail_kb < floor_kb and (mem_kb is None or mem_kb >= floor_kb):
        _say(f"WARNING: host memory available now is {avail_kb // 1048576} GiB (MemAvailable, capped by what the cgroup has left), "
             f"below the ~{floor_gib} GiB resident floor for {workers} workers{tier_s}")
        _say("         (another Merlin server or a large page-cache user?); the model load is expected to OOM.")
    try:
        st = os.statvfs("/dev/shm")
        shm_kb = st.f_bavail * st.f_frsize // 1024  # free, not the mount's total size
    except (OSError, AttributeError):
        shm_kb = None
    if shm_kb is not None and shm_kb < FN_SHM_MIN_KB:
        _say(f"WARNING: /dev/shm is {shm_kb // 1024} MB; the PLE transport needs >= 1 GB (docker run --shm-size=8g, compose shm_size: 8g)")
    return refusal


def banner(plan: Plan) -> None:
    """The resolved values a launch trusts from the environment, printed before anything runs."""
    env = plan.env
    host, port = effective_host_port(plan.args, plan.extra)
    print(
        f"merlin launch: {plan.checkpoint} profile {plan.profile.get('variant_of') or plan.profile['id']} [{plan.status}] "
        f"cards {env.get('CUDA_VISIBLE_DEVICES', '<unset>')} -> http://{host}:{port}; VENV={plan.venv} "
        f"TREE={plan.tree or '(installed engine)'} CACHE_ROOT={plan.cache_root}"
        + (f" SCALES={env['VLLM_QSA_KV_SCALES']}" if env.get("VLLM_QSA_KV_SCALES") else ""),
        file=sys.stderr,
    )


def check_scales_provenance(plan: Plan) -> None:
    """FP8 KV scales against the checkpoint and RoPE they run with. Recipe scales: see ``_check_recipe_scales``.
    Any scales with a ``<stem>.provenance.json`` beside them (tools/qsa_calibrate.py merge writes it): a warning when
    it names another checkpoint's config.json. Every scales file: a warning when the launch serves YaRN and the scales
    were not calibrated under YaRN (YaRN's attention scaling raises post-RoPE K past a native calibration; the
    magnitude is unmeasured)."""
    scales = plan.env.get("VLLM_QSA_KV_SCALES", "")
    if not scales:
        return
    d = Path(plan.ckpt)
    if plan.scales_recipe:
        _check_recipe_scales(plan)
    sp = Path(scales)
    prov_file = sp.with_name(sp.stem + ".provenance.json")
    try:
        prov = json.loads(prov_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        prov = None
    if not isinstance(prov, dict):
        prov = None
    cfg = d / "config.json"
    if prov and prov.get("config_sha256") and cfg.is_file() and prov["config_sha256"] != _sha256_file(cfg):
        print(f"WARN scales: {scales} were calibrated on {prov.get('model') or 'another checkpoint'} ({prov_file.name}), "
              f"whose config.json differs from {d}'s; calibrate this checkpoint with tools/qsa_calibrate.py", file=sys.stderr)
    # the recipes' scales are native-RoPE calibrations (262K); others say so only through their provenance file
    rope = prov.get("rope") if prov else "native" if plan.scales_recipe else None
    mml = [plan.extra[i + 1] for i, a in enumerate(plan.extra[:-1]) if a == "--max-model-len"]
    mml += [a.split("=", 1)[1] for a in plan.extra if a.startswith("--max-model-len=")]
    yarn = any("yarn" in a for a in plan.extra) or any((_small_whole(x) or 0) > 262144 for x in mml)
    if yarn and rope != "yarn":
        how = "without YaRN" if rope else "with unrecorded RoPE (no provenance file)"
        print(f"WARN scales: this launch serves YaRN but the FP8 KV scales {scales} were calibrated {how}; YaRN raises "
              f"post-RoPE K, so K can clip at FP8. Recalibrate under the same override: tools/qsa_calibrate.py serve "
              f"--max-model-len <N> -- --hf-overrides '<the YaRN override>'", file=sys.stderr)


def _check_recipe_scales(plan: Plan) -> None:
    """Recipe FP8 KV scales only on the checkpoint they were calibrated on. Quiet when the directory carries the
    recipe's own scales file (the release ships it with the weights) or a ``merlin setup`` verified marker for it; a
    warning when only config.json matches the recipe (same architecture, weights unverified: a fine-tune of it needs its
    own calibration); a refusal when config.json matches no revision of the recipe (a new model). SCALES= or a
    qsa_kv_scales.json next to the weights bypasses all of it (``scales_recipe`` is then None)."""
    name = plan.scales_recipe
    d = Path(plan.ckpt)
    scales = plan.env.get("VLLM_QSA_KV_SCALES", "")
    lst = _sha_list(plan.repo, name)
    fix = (f"calibrate this checkpoint with tools/qsa_calibrate.py and pass SCALES=<file> (or put it next to the weights "
           f"as {CKPT_SCALES})")
    if not lst:
        print(f"WARN scales: no checkpoint.sha256 for {name}; FP8 KV scales {scales} are not checked against {d}", file=sys.stderr)
        return
    own = {h for f, h in lst.items() if f.endswith(".json") and "scales" in f}
    for f in d.glob("*scales*.json"):
        if f.is_file() and _sha256_file(f) in own:
            return
    try:
        marker = json.loads((d / VERIFIED_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        marker = {}
    if isinstance(marker, dict) and str(marker.get("model", "")).split("+")[0] == name:  # <name>+<PLE table> variants
        return
    cfg = d / "config.json"
    if cfg.is_file() and lst.get("config.json") == _sha256_file(cfg):
        print(f"WARN scales: {d} has {name}'s config.json but its weights are not verified as {name} (no release scales "
              f"file, no merlin setup marker). The FP8 KV scales {scales} were calibrated on {name}; for a fine-tune of "
              f"it, {fix}.", file=sys.stderr)
        return
    raise LaunchError(
        f"uncalibrated: {d} is not a {name} checkpoint (its config.json differs from "
        f"{CHECKPOINTS[name][1]}/checkpoint.sha256), so the FP8 KV scales {scales}, calibrated on {name}, do not "
        f"apply to it. {fix[0].upper() + fix[1:]}.", 1)


def check_ple_table(ckpt: str, caller_env: dict[str, str], checkpoint: str | None = None) -> str:
    """The checkpoint's PLE table format: the shipped FP8, the INT6 g32 opt-in table (refused for a ``checkpoint`` name
    in PLE_INT6_UNQUALIFIED), or a test table the caller opted into by name."""
    # the key must sit INSIDE text_config (the engine reads its text config; a top-level copy does nothing)
    try:
        with open(os.path.join(ckpt, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        ple = cfg.get("text_config", {}).get("ple_embedding_dtype")
    except (OSError, ValueError, AttributeError):
        ple = None
    if ple == "float8_e4m3fn":
        return ple
    if ple in PLE_OPT_IN_TABLES:
        why = int6_unqualified_reason(checkpoint)
        if why:
            raise LaunchError(why, 1)
        print(f"PLE table: ple_embedding_dtype={ple} (the --ple int6 opt-in, not the FP8 default), host RAM "
              f"~{PLE_OPT_IN_TABLES[ple]} GiB for the table alone", file=sys.stderr)
        return ple
    if ple in PLE_TEST_TABLES and caller_env.get("MERLIN_PLE_TEST_TABLE") == ple:
        print(f"Test table: PLE table ple_embedding_dtype={ple} (MERLIN_PLE_TEST_TABLE); not a shipped format, "
              f"host RAM ~{PLE_TEST_TABLES[ple]} GiB for the table alone", file=sys.stderr)
        return ple
    hint = f" (a {ple} test table boots only with MERLIN_PLE_TEST_TABLE={ple})" if ple in PLE_TEST_TABLES else ""
    raise LaunchError(
        f"config.json lacks text_config.ple_embedding_dtype = float8_e4m3fn{hint}: run models/flash-next/make-e1-config.py {ckpt}", 1)


def check_remote_code(ckpt: str, pinned: Path) -> None:
    """Super VL loads its config and processor with --trust-remote-code, which runs the checkpoint's Python files:
    every one of them must match the reviewed hash list."""
    want: dict[str, str] = {}
    with open(pinned, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                digest, name = ln.split(None, 1)
                want[name.strip()] = digest
    for py in sorted(Path(ckpt).glob("*.py")):
        if want.get(py.name) != _sha256_file(py):
            raise LaunchError(
                f"{py} does not match the reviewed remote code ({pinned.name}); this checkpoint runs it with "
                "--trust-remote-code, so re-download the checkpoint before serving", 1)


def in_wsl(uname: str | None = None) -> bool:
    """WSL, by vLLM's own predicate (``vllm.platforms.interface.in_wsl``: "microsoft" anywhere in ``platform.uname()``)."""
    if uname is None:
        uname = " ".join(platform.uname())
    return "microsoft" in uname.lower()


def wsl_pin_memory(env: dict[str, str], family: str, uname: str | None = None) -> None:
    """On WSL2 vLLM turns pinned host memory off unless VLLM_WSL2_ENABLE_PIN_MEMORY=1, and the host-resident embedding
    tables need it: set it for the 27B, Super VL and Ornith (one embedding table per rank). Flash-Next can
    pin far more (its UVA PLE path), against WDDM's pinned-memory limit, so it is only told how. A caller's value wins."""
    if not in_wsl(uname) or "VLLM_WSL2_ENABLE_PIN_MEMORY" in env:
        return
    if family == "flash-next":
        print("note: WSL2: pinned host memory is off (vLLM's WSL default) and Flash-Next's host-resident embeddings need "
              "it; VLLM_WSL2_ENABLE_PIN_MEMORY=1 turns it on, but Windows limits pinned memory and Flash-Next can pin "
              "tens of GiB. Untested on WSL2", file=sys.stderr)
        return
    env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
    print("note: WSL2: VLLM_WSL2_ENABLE_PIN_MEMORY=1 (pinned host memory for the embedding table; WSL2 is untested for "
          "this engine)", file=sys.stderr)


def host_phase(plan: Plan) -> None:
    """Every check that touches the host; raises LaunchError instead of starting."""
    for n in plan.notes:
        print(n, file=sys.stderr)
    banner(plan)
    warn_retired(plan.caller_env)
    wsl_pin_memory(plan.env, plan.family)
    env, venv = plan.env, plan.venv
    check_cache_links(env["VLLM_CACHE_ROOT"])
    fam = plan.family
    ple = None
    if fam == "flash-next":
        tree = plan.tree or plan.repo.as_posix()
        if not os.path.isfile(f"{tree}/vllm/__init__.py"):
            msg = f"TREE={tree} is not a built tree (release/install-env.sh)"
            if os.path.exists(f"{plan.repo.as_posix()}/vllm-v2.5"):
                msg += f"\n       ({plan.repo.as_posix()}/vllm-v2.5 is an older build of the v2.5.1 source and is not used; see release/install-env.sh)"
            raise LaunchError(msg, 1)
        _verify_products(tree)
        if not os.access(f"{venv}/bin/python", os.X_OK):
            raise LaunchError(f"VENV={venv} has no bin/python (release/install-env.sh creates it)", 1)
        marker = plan.profile.get("checkpoint") and R.load_registry("flash-next")["checkpoints"][plan.profile["checkpoint"]].get("checkpoint_marker")
        if marker and not os.path.isfile(os.path.join(plan.ckpt, marker)):
            raise LaunchError(f"wrong checkpoint: missing {marker} in {plan.ckpt}", 1)
        ple = check_ple_table(plan.ckpt, plan.caller_env, plan.checkpoint)
        check_scales_provenance(plan)
        _check_shadow(tree)
        _toolchain(plan, full=True)
        _topology(plan)
    else:
        if not os.access(f"{venv}/bin/python", os.X_OK):
            raise LaunchError(f"VENV={venv} has no bin/python (release/install-env.sh creates it)", 1)
        if fam == "svl":
            check_remote_code(plan.ckpt, plan.repo / CHECKPOINTS["nemotron-super-vl"][1] / "remote-code.sha256")
        _toolchain(plan, full=False)
    for path, msg in plan.files:
        if not os.path.isfile(path):
            raise LaunchError(msg, 1)
    if fam == "flash-next":
        lay = plan.profile.get("layout") or {}
        # the KV host tier as the Flash-Next env builder passed it (set only when on, already checked)
        refusal = _host_memory_warnings(int(lay.get("tp", 1)) * int(lay.get("pp", 1)) if lay else 4,
                                        int(env.get(FN_KV_HOST_TIER) or 0), ple_table_gib(ple))
        if refusal and os.environ.get("MERLIN_SKIP_RAM_CHECK") != "1":  # =1 keeps the warning only
            raise LaunchError(refusal, 1)
    if fam in ("svl", "ornith"):
        os.makedirs(env["VLLM_CACHE_ROOT"], exist_ok=True)
        os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)


def effective_host_port(args: list[str], extra: list[str]) -> tuple[str, int]:
    """The host and port vllm will bind: the profile's flags, overridden by extra args (they come last and win)."""
    host, port = "127.0.0.1", 8000
    for seq in (args, extra):
        i = 0
        while i < len(seq):
            a = seq[i]
            if a in ("--host", "--port") and i + 1 < len(seq):
                if a == "--host":
                    host = seq[i + 1]
                else:
                    v = _small_whole(seq[i + 1])
                    port = port if v is None else v
                i += 2
                continue
            if a.startswith("--host="):
                host = a[7:]
            elif a.startswith("--port=") and _small_whole(a[7:]) is not None:
                port = int(a[7:])
            i += 1
    return host, port


def port_free(host: str, port: int) -> bool:
    """Whether ``host``:``port`` can be bound now, over IPv4 or IPv6 as the host resolves. False when the port is taken;
    LaunchError when the host is not an address this machine can bind."""
    import errno
    import socket

    h = host.strip("[]")
    try:
        if h in ("", "0.0.0.0"):
            family, addr = socket.AF_INET, ("", port)
        else:
            family, _, _, _, addr = socket.getaddrinfo(h, port, type=socket.SOCK_STREAM)[0]
    except (socket.gaierror, UnicodeError) as e:
        raise LaunchError(f"--host {host}: not a resolvable address ({e})", 2) from None
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(addr)
        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                return False
            raise LaunchError(f"--host {host} --port {port}: cannot bind ({e.strerror or e})", 2) from None
    return True


def print_hook(plan: Plan) -> None:
    env = plan.env
    _say("SERVE_PRINT_ENV_AND_EXIT:")
    _say(f"PROFILE={plan.profile.get('variant_of') or plan.profile['id']}"
         + (f"/kv-{R.KV_TAG[plan.profile['serve']['kv_cache_dtype']]}" if plan.profile.get("variant_of") else ""))
    _say(f"STATUS={plan.status}")
    if plan.p2p_mode:
        _say(f"P2P={plan.p2p_mode}")
    for var in HOOK_VARS:
        _say(f"{var}={env.get(var, '<unset>')}")
    _say(f"TOPO_LAUNCH_PREFIX={env.get('TOPO_LAUNCH_PREFIX', '')}")
    _say(f"TOPO_DECISION_JSON={env.get('TOPO_DECISION_JSON', '')}")
    _say("EXEC_ARGV_PREFIX:" + "".join(f" [{w}]" for w in plan.prefix))
    _say("ARGV:")
    for a in plan.args:
        _say(a)


def unauthenticated_exposure(host: str, env: dict[str, str], args: list[str], extra: list[str]) -> str:
    """A warning when the API listens beyond loopback with no key; empty otherwise. Inside a container the
    in-container bind (HOST=0.0.0.0) is expected and the published port decides the exposure, which the launcher
    cannot see, so the container wording names the port mapping instead of HOST."""
    if host in ("127.0.0.1", "::1", "localhost"):
        return ""
    if env.get("VLLM_API_KEY") or "--api-key" in args or any(a.startswith("--api-key") for a in extra):
        return ""
    if os.path.exists("/.dockerenv"):
        return (f"merlin: warning: no API key inside a container (HOST={host}): every client that can reach the "
                "published port is served without authentication; publish it on loopback only "
                "(docker run -p 127.0.0.1:8000:8000, as container/README.md shows) or set VLLM_API_KEY (bearer key)")
    return (f"merlin: warning: HOST={host} serves every client that can reach this host without authentication; "
            "set VLLM_API_KEY (bearer key) or HOST=127.0.0.1")


def run(plan: Plan) -> int:
    host_phase(plan)
    if os.environ.get("SERVE_PRINT_ENV_AND_EXIT", "0") == "1":
        print_hook(plan)
        return 0
    host, port = effective_host_port(plan.args, plan.extra)
    unauth = unauthenticated_exposure(host, plan.env, plan.args, plan.extra)
    if unauth:
        _say(unauth)
    if host not in ("127.0.0.1", "::1", "localhost"):
        _say(f"merlin: warning: HOST={host}: VLLM_API_KEY/--api-key does not guard /invocations "
             "(SageMaker generation) or ops routes (/metrics, /version, /health, /ping). "
             "Keep HOST=127.0.0.1 behind an authenticating reverse proxy or restrict network access; "
             "see docs/SETUP.md for API-key scope and client multimodal UUID limits.")
    # MERLIN_SKIP_PORT_CHECK=1 (exactly 1) skips this check: the escape hatch for a port probe that misreads a
    # free port (some container network setups); vLLM still fails cleanly if the port is really taken.
    if os.environ.get("MERLIN_SKIP_PORT_CHECK") != "1" and not port_free(host, port):
        raise LaunchError(
            f"port {port} on {host} is already in use (another server? the Flash-Next / Super VL / Ornith default is 8000, the 27B "
            f"default 8100); set PORT={port + 1} or stop the other server", 2)
    argv = plan.argv()
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(argv[0], argv, plan.env)
    return 0  # pragma: no cover


# --------------------------------------------------------------------------------------------
# Checkpoint identification (merlin serve ./ckpt with no name)
# --------------------------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_list(repo: Path, name: str) -> dict[str, str]:
    path = repo / CHECKPOINTS[name][1] / "checkpoint.sha256"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"([0-9a-fA-F]{64})\s+\*?(.+)$", ln.strip())
        if m:
            out[m.group(2).strip()] = m.group(1).lower()
    return out


def identify_checkpoint(ckpt: str, repo: Path | None = None) -> tuple[str, list[str]]:
    """Name the checkpoint in ``ckpt`` from config.json and the recipes' checkpoint.sha256 lists.

    Returns (name, notes). Small listed files are hashed to tell apart recipes whose config.json agrees; no safetensors
    shard is ever hashed. Raises LaunchError when the directory is not a Merlin checkpoint or stays ambiguous.
    """
    repo = Path(repo or REPO)
    d = Path(ckpt)
    notes: list[str] = []
    try:  # an INT6 PLE overlay (tools/merlin/ple_pack.py) names its model: its config.json and index are generated
        with open(d / ".merlin-overlay.json", encoding="utf-8") as f:
            ov = json.load(f).get("model")
        if ov in CHECKPOINTS:
            return ov, notes
    except (OSError, ValueError, AttributeError):
        pass
    try:
        with open(d / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise LaunchError(f"{ckpt}: no readable config.json ({e}); pass the checkpoint name (merlin list)", 2)
    names = {str(cfg.get("model_type", "")), *(cfg.get("architectures") or [])}
    # Load the shared stdlib-only parser without importing vllm and its GPU dependencies.
    family_py = repo / "vllm" / "flashnext_family.py"
    if not family_py.is_file():  # a recipe-only tree (no engine sources): use this checkout's parser
        family_py = REPO / "vllm" / "flashnext_family.py"
    try:
        aliases = runpy.run_path(str(family_py))["aliases"]()
    except ValueError as e:
        raise LaunchError(str(e), 2) from e
    names = {aliases.get(n, n).lower() for n in names}
    mt = " ".join(names)
    if "nemotron" in mt.lower():
        return "nemotron-super-vl", notes
    if "qwen3_5_moe" in mt.lower():
        # one Qwen3.5-MoE recipe (Ornith 1.5): name it only when config.json is the release's, byte for byte
        want = _sha_list(repo, "ornith1.5").get("config.json")
        if want and _sha256_file(d / "config.json") == want:
            return "ornith1.5", notes
        raise LaunchError(f"{ckpt}: a Qwen3.5-MoE checkpoint that is not the release's Ornith 1.5 pack (its config.json differs from "
                          "models/ornith1.5/checkpoint.sha256); pass the checkpoint name to serve it with that recipe (merlin list)", 2)
    if names & {"qwen4_exp", "qwen4_exp_text", "qwen4expforcausallm", "qwen4expforconditionalgeneration"}:
        family = "flash-next"
    elif any(n.startswith("qwen3_5") for n in names):
        family = "27b"
    else:
        raise LaunchError(f"{ckpt}: unknown model family (model_type {cfg.get('model_type')!r}, "
                          f"architectures {cfg.get('architectures')!r}); new model family: no Merlin recipe yet, "
                          "pass --profile/--model explicitly (merlin list)", 2)
    cands = [n for n, (model, _) in CHECKPOINTS.items() if model == family]
    if (d / "EDITED_TENSORS.txt").is_file() and "swift1.5-huihui" in cands:
        return "swift1.5-huihui", notes
    cands = [c for c in cands if c != "swift1.5-huihui"]
    # score by the listed small files that exist, then by their hashes
    lists = {c: _sha_list(repo, c) for c in cands}
    small = {c: {f: h for f, h in lst.items() if not f.endswith(".safetensors")} for c, lst in lists.items()}
    present = {c: {f for f in files if (d / f).is_file()} for c, files in small.items()}
    best = max((len(present[c]) / max(len(small[c]), 1), c) for c in cands)[0] if cands else 0
    top = [c for c in cands if small[c] and len(present[c]) / len(small[c]) == best and best > 0]
    if len(top) == 1:
        return top[0], notes
    if not top:
        raise LaunchError(f"{ckpt}: none of the recipes' checkpoint.sha256 lists matches this directory; pass the name (merlin list)", 2)
    hashed: dict[str, str] = {}
    for c in top:
        for fname in sorted(present[c]):
            if fname not in hashed:
                hashed[fname] = _sha256_file(d / fname)
    match = [c for c in top if all(small[c][f] == hashed[f] for f in present[c])]
    if len(match) == 1:
        return match[0], notes
    hint = "thinkingcap" in d.resolve().name.lower()
    if {"27b", "27b-thinkingcap"} <= set(top):
        pick = "27b-thinkingcap" if hint else "27b"
        notes.append(f"note: 27b and 27b-thinkingcap share every small file; taking {pick} from the directory name "
                     "(pass 27b or 27b-thinkingcap to be explicit)")
        return pick, notes
    raise LaunchError(f"{ckpt}: ambiguous between {', '.join(sorted(match or top))}; pass the checkpoint name", 2)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="launch.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=("flash-next", "27b", "svl", "ornith"), help="registry (default: the profile's)")
    ap.add_argument("--profile", required=True, help="profile id (tools/merlin/profiles/*.json)")
    ap.add_argument("--kv", choices=("fp8", "bf16"), help="KV variant of the profile")
    ap.add_argument("--checkpoint", choices=list(CHECKPOINTS), help="checkpoint name (default: the profile's)")
    ap.add_argument("--status",
                    choices=(R.LABEL_TESTED, R.LABEL_TESTING, R.LABEL_UNTESTED),
                    help="the label merlin serve evaluated on these cards (default: the registry row's)")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="[--] <checkpoint-dir> [extra vllm args]")
    a = ap.parse_args(argv)
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    model = a.model or ("27b" if a.profile.startswith("27b-") else "svl" if a.profile.startswith("svl-")
                        else "ornith" if a.profile.startswith("ornith-") else "flash-next")
    # <checkpoint-dir> positional, or MODEL= from the environment (then every argument is an extra vllm arg)
    if rest and not rest[0].startswith("-"):
        ckpt, extra = rest[0], rest[1:]
    else:
        env_model, env_dir = os.environ.get("MODEL"), os.environ.get("MERLIN_CHECKPOINT_DIR")
        if env_model and env_dir and os.path.realpath(env_model) != os.path.realpath(env_dir):
            # merlin serve verifies MERLIN_CHECKPOINT_DIR; serving another folder from MODEL would load what it did not check.
            # Compared resolved (a symlink alias of the same folder is the same folder; a ".." through a link is not).
            print(f"launch.py: MODEL={env_model} and MERLIN_CHECKPOINT_DIR={env_dir} name different folders; set one", file=sys.stderr)
            return 2
        ckpt, extra = (os.path.realpath(env_model) if env_model and env_dir else env_model or env_dir or ""), rest
    if not ckpt:
        print(f"usage: launch.py --profile {a.profile} -- <checkpoint-dir> [extra vllm args]   (or MODEL=<dir>)", file=sys.stderr)
        return 2
    try:
        plan = build(model, a.profile, ckpt, dict(os.environ), extra, a.kv, a.checkpoint, status=a.status)
        return run(plan)
    except LaunchError as e:
        if str(e):
            print(str(e), flush=True)
        return e.code
    except R.ProfileError as e:
        print(f"launch: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    R.safe_stdio()
    sys.exit(main())

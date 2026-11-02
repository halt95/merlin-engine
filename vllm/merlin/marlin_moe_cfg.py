# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Runtime override of the Marlin MoE block-size x thread-config.

``fused_marlin_moe`` picks ``block_size_m`` with a fixed heuristic and lets the
native launcher (``csrc/moe/marlin_moe_wna16/ops.cu``) pick the thread config.
The custom op already accepts ``thread_k`` / ``thread_n`` / ``blocks_per_sm``
but the engine never passes them. This module is the whole decision: given a
shape it returns either ``None`` (use the engine's own choice, byte-identical)
or a :class:`Decision` naming the block size and, per GEMM, a pinned thread
config chosen from kernels that are ALREADY COMPILED. No rebuild is involved.

Default OFF: with ``MERLIN_MARLIN_MOE_CFG`` unset or blank, :func:`enabled` is
False and :func:`select_override` returns ``None`` before it parses anything.

Syntax (rules separated by ``;``, first applicable match wins)::

    MERLIN_MARLIN_MOE_CFG = (
        "m=1024-2047,bm=32,g=64x128x2,ack=bitwise;m=4096-,bm=48,g=64x256x1,ack=bitwise"
    )

Keys of one rule (comma separated, each at most once):

* ``m``     token count the engine's block-size heuristic sees (``N``,
  ``LO-HI``, ``LO-`` or ``-HI``, inclusive; omitted = any). With expert
  parallelism this is the per-rank estimate ``ceil(M * E / global_experts)``,
  the same number the heuristic uses; without EP it is the batch row count.
* ``e`` ``topk`` ``k`` ``n``  exact-match filters on the local expert count,
  top-k, hidden size and per-expert intermediate size (omitted = any).
* ``bm``    block size M: 8, 16, 32, 48 or 64 (omitted = engine's choice).
* ``g1`` ``g2``  pinned ``THREAD_KxTHREAD_NxBLOCKS_PER_SM`` for the gate/up
  GEMM and the down GEMM; ``g`` sets both. Omitted = native auto choice.
* ``ack``   ``bitwise`` or ``nonexact``; see below.

Exactness. Marlin's per-row K reduction is fixed by the thread config, not by
``block_size_m`` (``use_full_k`` is always on, so the grid size does not enter).
But Python cannot see which thread config the native auto-selector picks for a
given ``bm`` (it depends on shared-memory fit), so no rule that differs from
the engine's choice is certifiable from here. The classes are:

* ``identity``   the decision equals the engine's choice: bit-identical.
* ``bm-only``    ``bm`` moves within 16/32/48/64, threads left to the native
  selector: structurally exact IF the native selector resolves the same thread
  tuple and blocks/SM for both block sizes, which is not guaranteed.
* ``thread-pin`` any ``g`` key: the K-reduction tree depends on the tuple, so
  identity is an empirical property of the shape, not of the code.
* ``non-exact``  switching to or from ``bm=8`` (different M layout).

Every rule must carry an ``ack``: ``bitwise`` states that a
microbench (the default and the rule's kernel tuple run on the same inputs
for each shape) showed ``torch.equal`` for this shape set; ``nonexact`` accepts differing bits. A
rule with no ack, or a ``bm=8`` rule without ``ack=nonexact``, is a boot error.
A ``non-exact`` outcome at run time under a bare ``bitwise`` ack (baseline was
8, or the rule moves to 8) skips the rule with a warning. Active rules log a
warning once per (rule, class) naming what the bit-identity rests on.

Pinned configs are checked here against the compiled set
(``csrc/quantization/marlin/generate_kernels.py``) and shape divisibility; a
rule that fails is skipped with a warning and the next rule is tried.
Shared-memory fit of a pinned tuple is checked only by the native launcher,
which raises on the first call.

Out of scope: ``batched_fused_marlin_moe`` (its block size is chosen
differently) and the ``MarlinExperts`` LoRA path are untouched.

Imports only the standard library and ``vllm.envs`` / ``vllm.logger`` so the
parsing and selection logic runs and is tested without a GPU.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_NAME = "MERLIN_MARLIN_MOE_CFG"

BLOCK_SIZES_M = (8, 16, 32, 48, 64)

# (thread_k, thread_n) kernels compiled by csrc/quantization/marlin/
# generate_kernels.py THREAD_CONFIGS: the 256-thread kernels exist only for
# (128,128) when thread_m_blocks <= 1 and only for (64,256) when it is > 1.
# num_threads is always thread_k * thread_n / 64, which ops.cu derives itself.
_COMPILED_SMALL_M = ((128, 128), (64, 128), (128, 64))  # bm 8, 16
_COMPILED_LARGE_M = ((64, 256), (64, 128), (128, 64))  # bm 32, 48, 64

BLOCKS_PER_SM = (1, 2, 3, 4)

EXACT_IDENTITY = "identity"
EXACT_BM_ONLY = "bm-only"
EXACT_THREAD_PIN = "thread-pin"
EXACT_NON_EXACT = "non-exact"

ACK_BITWISE = "bitwise"
ACK_NONEXACT = "nonexact"
_ACKS = (ACK_BITWISE, ACK_NONEXACT)

_KEYS = ("m", "e", "topk", "k", "n", "bm", "g", "g1", "g2", "ack")


def compiled_thread_configs(block_size_m: int) -> tuple[tuple[int, int], ...]:
    """(thread_k, thread_n) pairs with a compiled kernel for ``block_size_m``."""
    return _COMPILED_SMALL_M if block_size_m <= 16 else _COMPILED_LARGE_M


@dataclass(frozen=True)
class GemmCfg:
    """One pinned launch config for one GEMM."""

    thread_k: int
    thread_n: int
    blocks_per_sm: int

    def as_op_kwargs(self) -> dict[str, int]:
        return {
            "thread_k": self.thread_k,
            "thread_n": self.thread_n,
            "blocks_per_sm": self.blocks_per_sm,
        }

    def __str__(self) -> str:
        return f"{self.thread_k}x{self.thread_n}x{self.blocks_per_sm}"


def op_kwargs(gemm: GemmCfg | None) -> dict[str, int]:
    """Extra ``moe_wna16_marlin_gemm`` kwargs for a pinned GEMM; ``{}`` if not pinned.

    ``{}`` means the call is exactly the engine's own (the op's -1 defaults).
    """
    return {} if gemm is None else gemm.as_op_kwargs()


@dataclass(frozen=True)
class Rule:
    index: int
    text: str
    m_lo: int
    m_hi: int | None
    e: int | None
    topk: int | None
    k: int | None
    n: int | None
    bm: int | None
    g1: GemmCfg | None
    g2: GemmCfg | None
    ack: str


@dataclass(frozen=True)
class Decision:
    block_size_m: int
    gemm1: GemmCfg | None
    gemm2: GemmCfg | None
    rule: Rule
    exactness: str


def _int(rule: str, key: str, val: str) -> int:
    if not re.fullmatch(r"[0-9]+", val):
        raise ValueError(
            f"{ENV_NAME}: rule {rule!r}: {key}= needs a plain integer, got {val!r}"
        )
    return int(val)


def _parse_range(rule: str, val: str) -> tuple[int, int | None]:
    if re.fullmatch(r"[0-9]+", val):
        n = int(val)
        return n, n
    m = re.fullmatch(r"([0-9]*)-([0-9]*)", val)
    if m is None or (m.group(1) == "" and m.group(2) == ""):
        raise ValueError(
            f"{ENV_NAME}: rule {rule!r}: m= must be N, LO-HI, LO- or -HI, got {val!r}"
        )
    lo = int(m.group(1)) if m.group(1) else 0
    hi = int(m.group(2)) if m.group(2) else None
    if hi is not None and lo > hi:
        raise ValueError(f"{ENV_NAME}: rule {rule!r}: empty range m={val!r}")
    return lo, hi


def _parse_gemm(rule: str, key: str, val: str) -> GemmCfg:
    m = re.fullmatch(r"([0-9]+)x([0-9]+)x([0-9]+)", val)
    if m is None:
        raise ValueError(
            f"{ENV_NAME}: rule {rule!r}: {key}= must be THREAD_KxTHREAD_NxBLOCKS_PER_SM"
            f" (e.g. 64x128x2), got {val!r}"
        )
    cfg = GemmCfg(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if cfg.blocks_per_sm not in BLOCKS_PER_SM:
        raise ValueError(
            f"{ENV_NAME}: rule {rule!r}: {key}= blocks_per_sm {cfg.blocks_per_sm}"
            f" not in {BLOCKS_PER_SM}"
        )
    if (cfg.thread_k, cfg.thread_n) not in set(_COMPILED_SMALL_M) | set(
        _COMPILED_LARGE_M
    ):
        raise ValueError(
            f"{ENV_NAME}: rule {rule!r}: {key}= thread config"
            f" ({cfg.thread_k},{cfg.thread_n}) has no compiled kernel; compiled:"
            f" {sorted(set(_COMPILED_SMALL_M) | set(_COMPILED_LARGE_M))}"
        )
    return cfg


def _parse_rule(index: int, text: str) -> Rule:
    fields: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{ENV_NAME}: rule {text!r}: empty field")
        key, sep, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not val:
            raise ValueError(
                f"{ENV_NAME}: rule {text!r}: field {part!r} is not key=value"
            )
        if key not in _KEYS:
            raise ValueError(
                f"{ENV_NAME}: rule {text!r}: unknown key {key!r} (known: {_KEYS})"
            )
        if key in fields:
            raise ValueError(f"{ENV_NAME}: rule {text!r}: duplicate key {key!r}")
        fields[key] = val
    if "g" in fields and ("g1" in fields or "g2" in fields):
        raise ValueError(
            f"{ENV_NAME}: rule {text!r}: g= cannot be combined with g1=/g2="
        )

    m_lo, m_hi = (0, None)
    if "m" in fields:
        m_lo, m_hi = _parse_range(text, fields["m"])
    bm = None
    if "bm" in fields:
        bm = _int(text, "bm", fields["bm"])
        if bm not in BLOCK_SIZES_M:
            raise ValueError(
                f"{ENV_NAME}: rule {text!r}: bm={bm} not in {BLOCK_SIZES_M}"
            )
    g1 = g2 = None
    if "g" in fields:
        g1 = g2 = _parse_gemm(text, "g", fields["g"])
    if "g1" in fields:
        g1 = _parse_gemm(text, "g1", fields["g1"])
    if "g2" in fields:
        g2 = _parse_gemm(text, "g2", fields["g2"])
    if bm is None and g1 is None and g2 is None:
        raise ValueError(
            f"{ENV_NAME}: rule {text!r}: sets nothing (needs bm= and/or g/g1/g2)"
        )
    if bm is not None:
        for key, cfg in (("g1", g1), ("g2", g2)):
            if cfg is not None and (
                cfg.thread_k,
                cfg.thread_n,
            ) not in compiled_thread_configs(bm):
                raise ValueError(
                    f"{ENV_NAME}: rule {text!r}: {key}={cfg} has no compiled kernel at"
                    f" bm={bm}; compiled: {compiled_thread_configs(bm)}"
                )

    ack = fields.get("ack")
    if ack is None:
        raise ValueError(
            f"{ENV_NAME}: rule {text!r}: needs ack=bitwise (microbench torch.equal"
            " evidence exists for this shape set) or ack=nonexact; Python cannot"
            " certify any change from the engine's own choice"
        )
    if ack not in _ACKS:
        raise ValueError(
            f"{ENV_NAME}: rule {text!r}: ack= must be one of {_ACKS}, got {ack!r}"
        )
    if bm == 8 and ack != ACK_NONEXACT:
        raise ValueError(
            f"{ENV_NAME}: rule {text!r}: bm=8 uses a different M layout and is not"
            " bit-identical to bm>=16; it needs ack=nonexact"
        )

    def opt(key: str) -> int | None:
        return _int(text, key, fields[key]) if key in fields else None

    return Rule(
        index=index,
        text=text,
        m_lo=m_lo,
        m_hi=m_hi,
        e=opt("e"),
        topk=opt("topk"),
        k=opt("k"),
        n=opt("n"),
        bm=bm,
        g1=g1,
        g2=g2,
        ack=ack,
    )


def parse_rules(raw: str | None) -> tuple[Rule, ...]:
    """Parse a ``MERLIN_MARLIN_MOE_CFG`` value. Unset or blank = no rules.

    Raises ``ValueError`` on anything malformed: a switch that selects kernels
    must not be satisfiable by a spelling nobody meant.
    """
    if raw is None or not raw.strip():
        return ()
    rules = []
    for text in raw.split(";"):
        text = text.strip()
        if not text:
            continue
        rules.append(_parse_rule(len(rules), text))
    if not rules:
        raise ValueError(f"{ENV_NAME}: {raw!r} contains no rule")
    return tuple(rules)


@functools.lru_cache(maxsize=16)
def _parse_cached(raw: str) -> tuple[Rule, ...]:
    return parse_rules(raw)


def _raw() -> str | None:
    return envs.MERLIN_MARLIN_MOE_CFG


def enabled() -> bool:
    """True iff the env var is set non-blank. The only work on the flag-off path."""
    raw = _raw()
    return bool(raw and raw.strip())


def active_rules() -> tuple[Rule, ...]:
    """The parsed rules (cached per raw value); raises ValueError if malformed."""
    raw = _raw()
    if not raw or not raw.strip():
        return ()
    return _parse_cached(raw)


def classify(baseline_bm: int, block_size_m: int, pinned: bool) -> str:
    """Exactness class of moving from the engine's block size to ``block_size_m``."""
    if (baseline_bm == 8) != (block_size_m == 8):
        return EXACT_NON_EXACT
    if pinned:
        return EXACT_THREAD_PIN
    if block_size_m != baseline_bm:
        return EXACT_BM_ONLY
    return EXACT_IDENTITY


_warned: set[tuple[object, ...]] = set()


def _warn_once(key: tuple[object, ...], msg: str, *args: object) -> None:
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(msg, *args)


_SHAPE_LOG_CAP = 64
_shape_decisions: dict[tuple[int, ...], str] = {}


def _record_shape(shape: tuple[int, ...], outcome: str, skipped: list[str]) -> None:
    note = outcome + (" after " + "; ".join(skipped) if skipped else "")
    if _shape_decisions.get(shape) == note:
        return
    if len(_shape_decisions) >= _SHAPE_LOG_CAP and shape not in _shape_decisions:
        return
    _shape_decisions[shape] = note
    logger.info(
        "%s applied for M=%d E=%d topk=%d K=%d N=%d: %s", ENV_NAME, *shape, note
    )


def _matches(rule: Rule, *, m: int, e: int, topk: int, k: int, n: int) -> bool:
    if m < rule.m_lo or (rule.m_hi is not None and m > rule.m_hi):
        return False
    return all(
        want is None or want == have
        for want, have in ((rule.e, e), (rule.topk, topk), (rule.k, k), (rule.n, n))
    )


def _gemm_problem_ok(
    cfg: GemmCfg | None, bm: int, size_k: int, size_n: int
) -> str | None:
    """None if ``cfg`` is launchable for this GEMM, else the reason it is not."""
    if cfg is None:
        return None
    if (cfg.thread_k, cfg.thread_n) not in compiled_thread_configs(bm):
        return f"{cfg} has no compiled kernel at bm={bm}"
    if size_k % cfg.thread_k:
        return f"size_k={size_k} not divisible by thread_k={cfg.thread_k}"
    if size_n % cfg.thread_n:
        return f"size_n={size_n} not divisible by thread_n={cfg.thread_n}"
    return None


def select_override(
    *,
    M: int,
    topk: int,
    E: int,
    K: int,
    N: int,
    w13_shards: int,
    baseline_bm: int,
    min_bm: int = 8,
    rules: tuple[Rule, ...] | None = None,
) -> Decision | None:
    """The per-shape decision. ``None`` = keep the engine's own choice.

    ``M`` is the token count the heuristic used, ``baseline_bm`` the block size
    it (and the int8 lift) produced, ``min_bm`` the smallest block size the
    input dtype allows (16 for 1-byte activations), ``N`` the per-expert
    intermediate size and ``w13_shards`` 2 for gated activations else 1 (the
    first GEMM is ``size_n = w13_shards * N, size_k = K``, the second
    ``size_n = K, size_k = N``).

    ``rules`` overrides the rule table; production leaves it ``None`` and the
    parsed env value is used. With the env unset this returns ``None``
    immediately.
    """
    if rules is None:
        if not enabled():
            return None
        rules = active_rules()
    shape = (M, E, topk, K, N)
    skipped: list[str] = []
    for rule in rules:
        if not _matches(rule, m=M, e=E, topk=topk, k=K, n=N):
            continue
        bm = rule.bm if rule.bm is not None else baseline_bm
        pinned = rule.g1 is not None or rule.g2 is not None
        klass = classify(baseline_bm, bm, pinned)

        if bm < min_bm:
            skipped.append(f"rule#{rule.index} skipped[bm<{min_bm}]")
            _warn_once(
                (rule.index, "min_bm", min_bm),
                "%s rule #%d %r skipped: bm=%d is below the %d this input dtype allows",
                ENV_NAME,
                rule.index,
                rule.text,
                bm,
                min_bm,
            )
            continue
        if klass == EXACT_NON_EXACT and rule.ack != ACK_NONEXACT:
            skipped.append(f"rule#{rule.index} skipped[needs ack=nonexact]")
            _warn_once(
                (rule.index, "nonexact-skip", baseline_bm),
                "%s rule #%d %r skipped: engine bm=%d -> rule bm=%d crosses the bm=8"
                " layout and is not bit-identical; the rule has ack=%s, needs"
                " ack=nonexact",
                ENV_NAME,
                rule.index,
                rule.text,
                baseline_bm,
                bm,
                rule.ack,
            )
            continue
        reason = _gemm_problem_ok(rule.g1, bm, K, w13_shards * N) or _gemm_problem_ok(
            rule.g2, bm, N, K
        )
        if reason is not None:
            skipped.append(f"rule#{rule.index} skipped[{reason}]")
            _warn_once(
                (rule.index, "invalid", M, E, K, N),
                "%s rule #%d %r skipped for M=%d E=%d K=%d N=%d: %s",
                ENV_NAME,
                rule.index,
                rule.text,
                M,
                E,
                K,
                N,
                reason,
            )
            continue
        if klass == EXACT_IDENTITY:
            _record_shape(shape, f"rule#{rule.index} identity (engine-default)", skipped)
            return None
        _warn_once(
            (rule.index, klass),
            "%s rule #%d %r is ACTIVE, class=%s, ack=%s. Bit-identity with the"
            " engine's own choice is %s",
            ENV_NAME,
            rule.index,
            rule.text,
            klass,
            rule.ack,
            "NOT expected (ack=nonexact)"
            if rule.ack == ACK_NONEXACT or klass == EXACT_NON_EXACT
            else "asserted by the microbench torch.equal evidence behind ack=bitwise,"
            " not by the kernel structure",
        )
        _record_shape(shape, f"rule#{rule.index} bm={bm} class={klass}", skipped)
        return Decision(
            block_size_m=bm, gemm1=rule.g1, gemm2=rule.g2, rule=rule, exactness=klass
        )
    if rules:
        _record_shape(shape, "engine-default", skipped)
    return None

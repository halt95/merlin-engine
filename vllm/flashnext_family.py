# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One family-name table for the Flash-Next (Qwen4Exp) model family.

Every gate that checks for ``"Qwen4ExpForCausalLM"`` / ``"qwen4_exp"`` reads
this table rather than a literal, so a renamed upstream class is one edit here,
or one env entry with no code change::

    VLLM_FLASHNEXT_FAMILY_ALIASES="Qwen4ForCausalLM=Qwen4ExpForCausalLM,qwen4=qwen4_exp"

Each ``alias=canonical`` entry makes ``alias`` behave exactly like ``canonical``
at every site (model registry, config-class map, RecoverSSM allowlist, PLE
n-gram layer fields, the speculative MTP rewrite, the graph-profile guards and the PLE
pull-transport gate).  Unset, the resolved sets are exactly the historical
literals, so Flash-Next behaviour is unchanged. Set the env before importing
vLLM: HF config and model registries install aliases once at startup.

This module is stdlib-only on purpose: it is imported by the config layer, the
model registry and the worker, and must not create import cycles.
"""

import os
from collections.abc import Iterable, Mapping
from typing import TypeVar

ENV_ALIASES = "VLLM_FLASHNEXT_FAMILY_ALIASES"

#: Role -> canonical architecture (class) name.
CANONICAL_ARCH: Mapping[str, str] = {
    "causal_lm": "Qwen4ExpForCausalLM",
    "conditional_generation": "Qwen4ExpForConditionalGeneration",
    "mtp": "Qwen4ExpMTP",
}

#: Architectures that are a target model (not the MTP drafter).
TARGET_ROLES = ("causal_lm", "conditional_generation")
ALL_ROLES = ("causal_lm", "conditional_generation", "mtp")

#: Canonical HF ``model_type`` values of the family (config / text config).
CANONICAL_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")
#: The model_type the speculative config rewrites to for the native MTP head.
MTP_MODEL_TYPE = "qwen4_exp_mtp"
#: model_type -> HF config class name (resolved lazily by transformers_utils).
MODEL_TYPE_CONFIG_CLASS: Mapping[str, str] = {
    "qwen4_exp": "Qwen4ExpConfig",
    "qwen4_exp_text": "Qwen4ExpTextConfig",
}

_V = TypeVar("_V")


def _parse(raw: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    known = set(CANONICAL_ARCH.values()) | set(CANONICAL_MODEL_TYPES)
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        alias, sep, canonical = (part.strip() for part in item.partition("="))
        if not sep or not alias or not canonical:
            raise ValueError(
                f"{ENV_ALIASES} entries must be alias=canonical, got {item!r}"
            )
        if canonical not in known:
            raise ValueError(
                f"{ENV_ALIASES}: {canonical!r} is not a Flash-Next family name; "
                f"expected one of {sorted(known)}"
            )
        if alias in known:
            raise ValueError(
                f"{ENV_ALIASES}: alias {alias!r} is already a canonical name"
            )
        aliases[alias] = canonical
    return aliases


_cache: tuple[str, dict[str, str]] | None = None


def aliases() -> dict[str, str]:
    """alias -> canonical name, parsed from the env (empty when unset)."""
    global _cache
    raw = os.environ.get(ENV_ALIASES, "")
    if _cache is None or _cache[0] != raw:
        _cache = (raw, _parse(raw))
    return dict(_cache[1])


def archs(roles: Iterable[str] = ALL_ROLES) -> tuple[str, ...]:
    """Canonical + aliased architecture names for the given roles."""
    canon = [CANONICAL_ARCH[role] for role in roles]
    extra = [a for a, c in aliases().items() if c in canon]
    return (*canon, *extra)


def target_archs() -> tuple[str, ...]:
    return archs(TARGET_ROLES)


def model_types() -> frozenset[str]:
    """Canonical + aliased target model_type values."""
    extra = {a for a, c in aliases().items() if c in CANONICAL_MODEL_TYPES}
    return frozenset((*CANONICAL_MODEL_TYPES, *extra))


def expand_by_role(values: Mapping[str, _V]) -> dict[str, _V]:
    """{role: value} -> {arch name: value} for the canonical and aliased names."""
    out: dict[str, _V] = {}
    for role, value in values.items():
        for name in archs((role,)):
            out[name] = value
    return out


def expand_model_types(values: Mapping[str, _V]) -> dict[str, _V]:
    """{canonical model_type: value} -> also keyed by its aliases."""
    out: dict[str, _V] = dict(values)
    for alias, canonical in aliases().items():
        if canonical in values:
            out[alias] = values[canonical]
    return out

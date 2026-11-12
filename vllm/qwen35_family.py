# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""One name table for the Qwen3.5 lineage (dense: Qwen3.8-27B; MoE: Ornith 1.5).

Every Merlin gate that matches ``"Qwen3_5ForCausalLM"`` and friends reads this
table, not a literal tuple: the 27B sm_86 series patches and the
Inductor reduction pin (``v1/qwen35_series.py``), the RecoverSSM allow-list and
its PP rule (``config/vllm.py``), the host-embed table (``merlin/envs.py``,
``v1/worker/gpu/model_runner.py``), the GDN Triton warm-up and the speculative
MTP rewrite (``config/speculative.py``).

The table has three layers:

* the canonical Qwen3.5 names (today's checkpoints);
* ``CANDIDATE_*``: plausible Qwen 4 names on the same lineage (``Qwen4For*``,
  ``qwen4``...). They join the gates only. They are NOT added to the model
  registry or the HF config map, so a checkpoint with one of these names still
  needs the engine (or an alias below) to know its class;
* ``MERLIN_QWEN35_ARCH_ALIASES``, a comma-separated env list read at import::

      MERLIN_QWEN35_ARCH_ALIASES="Qwen4ForConditionalGeneration=Qwen3_5ForConditionalGeneration,qwen4=qwen3_5,qwen4_text=qwen3_5_text"

  ``alias=canonical`` makes ``alias`` behave exactly like ``canonical``: it joins
  the canonical name's gates, an architecture alias is also served by the
  canonical name's model class (model registry + config hook map) and a
  model_type alias is loaded with the canonical HF config class and renamed to
  the canonical model_type after loading (the Qwen3.5 model code dispatches on
  it). A bare ``Name`` joins the gates only (dense, or MoE when the name
  contains "moe"); use it when the engine already registers the class. A
  ``-Name`` entry drops a default candidate. Names are architectures when they
  contain an upper-case letter and model_types when they are all lower case.

  Refused: an alias that already names an architecture the engine registers
  (model registry, config-hook map) or a model_type HF already maps, the same
  alias with two different targets, a model_type alias whose ``_text`` suffix
  differs from its target's, a name that is both added and dropped, and the
  MTP rewrite's ``qwen3_5_mtp``.

A name that belongs to the Flash-Next family (``vllm/flashnext_family.py``, its
``VLLM_FLASHNEXT_FAMILY_ALIASES`` included) is never a Qwen3.5 name: the default
candidates drop it, and an explicit entry naming it is refused.

Unset, the canonical sets are exactly the historical literals; the candidates
match no checkpoint shipped today. Set the env before importing vLLM: the
config layer and the model registry resolve the sets once at import.

Stdlib-only apart from the Flash-Next table (itself stdlib-only), like
``flashnext_family``: imported by the config layer, the model registry and the
worker.
"""

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from vllm import flashnext_family

ENV_ALIASES = "MERLIN_QWEN35_ARCH_ALIASES"

DENSE = "dense"
MOE = "moe"

#: Canonical architecture -> group, in the historical tuple order.
CANONICAL_ARCHS: Mapping[str, str] = {
    "Qwen3_5ForCausalLM": DENSE,
    "Qwen3_5ForConditionalGeneration": DENSE,
    "Qwen3_5MoeForCausalLM": MOE,
    "Qwen3_5MoeForConditionalGeneration": MOE,
}
#: Canonical HF model_type -> group (top-level and text configs).
CANONICAL_MODEL_TYPES: Mapping[str, str] = {
    "qwen3_5": DENSE,
    "qwen3_5_text": DENSE,
    "qwen3_5_moe": MOE,
    "qwen3_5_moe_text": MOE,
}
#: model_type -> transformers config class name (resolved lazily).
MODEL_TYPE_CONFIG_CLASS: Mapping[str, str] = {
    "qwen3_5": "Qwen3_5Config",
    "qwen3_5_text": "Qwen3_5TextConfig",
    "qwen3_5_moe": "Qwen3_5MoeConfig",
    "qwen3_5_moe_text": "Qwen3_5MoeTextConfig",
}

#: Plausible Qwen 4 names on the Qwen3.5 lineage (gates only, see above).
CANDIDATE_ARCHS: Mapping[str, str] = {
    "Qwen4ForCausalLM": DENSE,
    "Qwen4ForConditionalGeneration": DENSE,
    "Qwen4MoeForCausalLM": MOE,
    "Qwen4MoeForConditionalGeneration": MOE,
}
CANDIDATE_MODEL_TYPES: Mapping[str, str] = {
    "qwen4": DENSE,
    "qwen4_text": DENSE,
    "qwen4_moe": MOE,
    "qwen4_moe_text": MOE,
}

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_V = TypeVar("_V")


def _is_model_type(name: str) -> bool:
    return name == name.lower()


def _is_text(model_type: str) -> bool:
    return model_type.endswith("_text")


#: model_types an alias may not take: the speculative MTP rewrite's output.
_RESERVED_MODEL_TYPES = frozenset({"qwen3_5_mtp"})


def _group_of(name: str) -> str:
    return MOE if "moe" in name.lower() else DENSE


def _flashnext_names() -> frozenset[str]:
    return frozenset(
        (
            *flashnext_family.archs(),
            *flashnext_family.model_types(),
            flashnext_family.MTP_MODEL_TYPE,
        )
    )


class _Table:
    def __init__(self, raw: str) -> None:
        canonical = {**CANONICAL_ARCHS, **CANONICAL_MODEL_TYPES}
        candidates = {**CANDIDATE_ARCHS, **CANDIDATE_MODEL_TYPES}
        flashnext = _flashnext_names()
        self.aliases: dict[str, str] = {}  # alias -> canonical
        extra: dict[str, str] = {}  # bare name -> group
        removed: set[str] = set()
        for item in raw.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            drop = item.startswith("-")
            body = item[1:] if drop else item  # exactly one "-"; "--Name" is refused
            name, sep, target = (p.strip() for p in body.partition("="))
            if not _NAME.fullmatch(name) or (sep and not _NAME.fullmatch(target)):
                raise ValueError(
                    f"{ENV_ALIASES} entries are Name, alias=canonical or -Name, "
                    f"got {item!r}"
                )
            if drop:
                if sep or name not in candidates:
                    raise ValueError(
                        f"{ENV_ALIASES}: -{name}: only a default candidate can be "
                        f"dropped, one of {sorted(candidates)}"
                    )
                removed.add(name)
                continue
            if name in canonical:
                raise ValueError(
                    f"{ENV_ALIASES}: {name!r} is already a canonical Qwen3.5 name"
                )
            if name in _RESERVED_MODEL_TYPES:
                raise ValueError(
                    f"{ENV_ALIASES}: {name!r} is the model_type the speculative "
                    "MTP rewrite produces; it cannot be an alias"
                )
            if name in flashnext:
                raise ValueError(
                    f"{ENV_ALIASES}: {name!r} is a Flash-Next family name "
                    f"({flashnext_family.ENV_ALIASES}); it cannot also be Qwen3.5"
                )
            if sep:
                if target not in canonical:
                    raise ValueError(
                        f"{ENV_ALIASES}: {target!r} is not a canonical Qwen3.5 "
                        f"name; expected one of {sorted(canonical)}"
                    )
                if _is_model_type(name) != _is_model_type(target):
                    raise ValueError(
                        f"{ENV_ALIASES}: {item!r} maps an architecture onto a "
                        "model_type (or back); alias like with like"
                    )
                if _is_model_type(name) and _is_text(name) != _is_text(target):
                    raise ValueError(
                        f"{ENV_ALIASES}: {item!r} maps a top-level model_type onto "
                        "a text-config one (or back); a text-config alias ends in "
                        "'_text' like its target"
                    )
                if self.aliases.get(name, target) != target:
                    raise ValueError(
                        f"{ENV_ALIASES}: {name!r} is aliased twice, to "
                        f"{self.aliases[name]!r} and {target!r}"
                    )
                self.aliases[name] = target
                extra.pop(name, None)
            elif name not in self.aliases:
                extra[name] = _group_of(name)
        contradicted = sorted(removed & (set(self.aliases) | set(extra)))
        if contradicted:
            raise ValueError(
                f"{ENV_ALIASES}: {', '.join(contradicted)} both added and dropped "
                "(-Name); keep one of the two entries"
            )
        groups: dict[str, str] = dict(canonical)
        for name, group in candidates.items():
            if name not in removed and name not in flashnext:
                groups[name] = group
        for alias, target in self.aliases.items():
            groups[alias] = canonical[target]
        groups.update(extra)
        self.groups = groups

    def names(self, group: str | None, model_types: bool) -> tuple[str, ...]:
        return tuple(
            n
            for n, g in self.groups.items()
            if (group is None or g == group) and _is_model_type(n) == model_types
        )


_cache: tuple[str, _Table] | None = None


def _table() -> _Table:
    global _cache
    raw = os.environ.get(ENV_ALIASES, "")
    flashnext_raw = os.environ.get(flashnext_family.ENV_ALIASES, "")
    key = f"{raw}\0{flashnext_raw}"
    if _cache is None or _cache[0] != key:
        _cache = (key, _Table(raw))
    return _cache[1]


def aliases() -> dict[str, str]:
    """Map alias -> canonical name for the ``alias=canonical`` entries."""
    return dict(_table().aliases)


def dense_archs() -> tuple[str, ...]:
    """Dense Qwen3.5-lineage architectures: canonical first, then the rest."""
    return _table().names(DENSE, model_types=False)


def moe_archs() -> tuple[str, ...]:
    """Qwen3.5 MoE-lineage architectures: canonical first, then the rest."""
    return _table().names(MOE, model_types=False)


def gdn_archs() -> tuple[str, ...]:
    """Every Qwen3.5-lineage architecture (dense, then MoE)."""
    return (*dense_archs(), *moe_archs())


def model_types() -> tuple[str, ...]:
    """Every Qwen3.5-lineage model_type (canonical first)."""
    return _table().names(None, model_types=True)


def moe_model_types() -> tuple[str, ...]:
    return _table().names(MOE, model_types=True)


def is_dense_arch(architecture: Any) -> bool:
    return architecture in dense_archs()


def is_moe_arch(architecture: Any) -> bool:
    return architecture in moe_archs()


def expand_aliases(values: Mapping[str, _V]) -> dict[str, _V]:
    """{canonical name: value} -> {alias: value} for the ``alias=canonical``
    entries whose canonical name is a key (the canonical keys themselves are not
    repeated: the caller keeps its own literal entries)."""
    return {a: values[c] for a, c in _table().aliases.items() if c in values}


def expand_by_suffix(values: Mapping[str, _V]) -> dict[str, _V]:
    """{class-name suffix: value} -> {alias: value} for the architecture aliases,
    keyed by the suffix of their canonical name (``"ForCausalLM"`` /
    ``"ForConditionalGeneration"``)."""
    return {
        a: v
        for a, c in _table().aliases.items()
        for suffix, v in values.items()
        if c in CANONICAL_ARCHS and c.endswith(suffix)
    }


#: Module of the canonical Qwen3.5 model classes under vllm/model_executor/models.
MODEL_MODULE = "qwen3_5"


def registry_aliases(suffix: str) -> dict[str, tuple[str, str]]:
    """Model-registry entries ``alias -> (module, canonical class)`` for the
    architecture aliases whose canonical class name ends in ``suffix``
    (``"ForCausalLM"`` or ``"ForConditionalGeneration"``)."""
    return expand_aliases(
        {c: (MODEL_MODULE, c) for c in CANONICAL_ARCHS if c.endswith(suffix)}
    )


def canonical_model_type(model_type: Any) -> Any:
    """The canonical model_type for an aliased one; anything else unchanged."""
    if not isinstance(model_type, str):
        return model_type
    target = _table().aliases.get(model_type)
    return target if target in CANONICAL_MODEL_TYPES else model_type


#: Canonical multimodal model_type -> the model_type of its text config.
_TEXT_MODEL_TYPE = {"qwen3_5": "qwen3_5_text", "qwen3_5_moe": "qwen3_5_moe_text"}


def canonicalize_config(config: Any) -> Any:
    """Rename an aliased ``model_type`` to the canonical one after the HF config
    has been loaded and overridden; the Qwen3.5 model code dispatches on it.

    The parent and its text config are handled independently: each one with an
    alias of its own is renamed to its target (so a canonical ``qwen3_5`` parent
    with a ``qwen4_text`` text config and ``qwen4_text=qwen3_5_text`` ends as
    ``qwen3_5`` / ``qwen3_5_text``). When the parent was renamed and its text
    config has no alias, the text config follows the parent (``qwen4=qwen3_5``
    alone also renames a ``qwen4_text`` text config to ``qwen3_5_text``). A
    no-op without ``alias=canonical`` model_type entries, so today's configs are
    never touched. Idempotent: it runs after loading and again after every
    hf_overrides layer, and never undoes the speculative MTP rewrite
    (``qwen3_5_mtp`` can not be an alias).
    """
    if not _table().aliases:
        return config
    current = getattr(config, "model_type", None)
    target = canonical_model_type(current)
    renamed = target != current
    if renamed:
        config.model_type = target
    text = getattr(config, "text_config", None)
    if text is not None and text is not config:
        text_type = getattr(text, "model_type", None)
        text_target = canonical_model_type(text_type)
        if (
            text_target == text_type
            and renamed
            and text_type not in CANONICAL_MODEL_TYPES
        ):
            text_target = _TEXT_MODEL_TYPE.get(target, text_type)
        if text_target != text_type:
            text.model_type = text_target
    return config


def refuse_registered(names: Iterable[str], where: str, *registered: Mapping) -> None:
    """Refuse an ``alias=canonical`` architecture alias whose name is already a key
    of one of ``registered`` (the maps before the aliases join them): installing
    it would silently re-point (or be shadowed by) a model the engine serves."""
    taken = sorted({n for n in names if any(n in m for m in registered)})
    if taken:
        raise ValueError(
            f"{ENV_ALIASES}: {', '.join(taken)} already names an architecture in "
            f"{where}; an alias must be a new name (use a bare Name to add a "
            "registered architecture to the Qwen3.5 gates)"
        )


def install_registry_aliases(
    text_models: dict, multimodal_models: dict, *registered: Mapping
) -> None:
    """Add the architecture aliases to the model registry's text-generation and
    multimodal maps (``alias -> (module, canonical class)``), after refusing any
    alias that a registry map in ``registered`` (or either target) already holds."""
    causal = registry_aliases("ForCausalLM")
    conditional = registry_aliases("ForConditionalGeneration")
    refuse_registered(
        (*causal, *conditional),
        "the model registry",
        text_models,
        multimodal_models,
        *registered,
    )
    text_models.update(causal)
    multimodal_models.update(conditional)

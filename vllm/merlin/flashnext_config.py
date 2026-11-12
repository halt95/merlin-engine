# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Install startup-only model-type aliases in HF's config mapping.

Two alias tables feed it: the Flash-Next family (``VLLM_FLASHNEXT_FAMILY_ALIASES``,
vllm/flashnext_family.py) and the Qwen3.5 lineage (``MERLIN_QWEN35_ARCH_ALIASES``,
vllm/qwen35_family.py). Both are validated before either installs anything.
"""

from collections.abc import Iterable, Mapping

from vllm import flashnext_family, qwen35_family


def _model_type_aliases(
    aliases: Mapping[str, str], config_class: Mapping[str, str]
) -> dict[str, str]:
    """Map each alias that names a model_type to its HF config class name."""
    return {a: config_class[c] for a, c in aliases.items() if c in config_class}


def register_config_aliases(reserved_model_types: Iterable[str] = ()) -> None:
    """Register aliases without changing canonical config classes.

    Set the env before importing vLLM's config/model registries. Like the model
    registry's architecture aliases, these registrations last for the process.
    """
    expanded = flashnext_family.expand_model_types(
        flashnext_family.MODEL_TYPE_CONFIG_CLASS
    )
    tables = (
        (
            flashnext_family.ENV_ALIASES,
            {
                alias: expanded[alias]
                for alias in flashnext_family.aliases()
                if alias in expanded
            },
        ),
        (
            qwen35_family.ENV_ALIASES,
            _model_type_aliases(
                qwen35_family.aliases(), qwen35_family.MODEL_TYPE_CONFIG_CLASS
            ),
        ),
    )
    if not any(model_aliases for _, model_aliases in tables):
        return

    import transformers
    from transformers import CONFIG_MAPPING

    pending = {}
    for env_name, model_aliases in tables:
        for alias, class_name in model_aliases.items():
            config_class = getattr(transformers, class_name)
            if alias in reserved_model_types or (
                alias in CONFIG_MAPPING and CONFIG_MAPPING[alias] is not config_class
            ):
                raise ValueError(
                    f"{env_name}: model-type alias {alias!r} "
                    "is already registered for another config"
                )
            pending[alias] = config_class

    # AutoConfig.register requires alias == config_class.model_type. Register
    # with its underlying mapping instead: mutating the shared class would
    # corrupt canonical loads and aliases pointing to the same class.
    # Validate the whole batch first so a refused alias installs nothing.
    for alias, config_class in pending.items():
        CONFIG_MAPPING.register(alias, config_class)


def canonicalize_qwen35_model_types(config):
    """After loading: an aliased Qwen3.5-lineage model_type becomes the canonical
    one (the Qwen3.5 model code dispatches on it). No-op without such aliases."""
    return qwen35_family.canonicalize_config(config)

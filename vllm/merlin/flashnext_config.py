# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Install startup-only Flash-Next model-type aliases in HF's config mapping."""

from collections.abc import Iterable

from vllm import flashnext_family


def register_config_aliases(reserved_model_types: Iterable[str] = ()) -> None:
    """Register aliases without changing canonical config classes.

    Set the env before importing vLLM's config/model registries. Like the model
    registry's architecture aliases, these registrations last for the process.
    """
    expanded = flashnext_family.expand_model_types(
        flashnext_family.MODEL_TYPE_CONFIG_CLASS
    )
    model_aliases = {
        alias: expanded[alias]
        for alias in flashnext_family.aliases()
        if alias in expanded
    }
    if not model_aliases:
        return

    import transformers
    from transformers import CONFIG_MAPPING

    pending = {}
    for alias, class_name in model_aliases.items():
        config_class = getattr(transformers, class_name)
        if alias in reserved_model_types or (
            alias in CONFIG_MAPPING and CONFIG_MAPPING[alias] is not config_class
        ):
            raise ValueError(
                f"{flashnext_family.ENV_ALIASES}: model-type alias {alias!r} "
                "is already registered for another config"
            )
        pending[alias] = config_class

    # AutoConfig.register requires alias == config_class.model_type. Register
    # with its underlying mapping instead: mutating the shared class would
    # corrupt canonical loads and aliases pointing to the same class.
    # Validate the whole batch first so a refused alias installs nothing.
    for alias, config_class in pending.items():
        CONFIG_MAPPING.register(alias, config_class)

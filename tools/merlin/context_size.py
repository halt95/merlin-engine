# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Context selection for the Merlin launcher.

No engine imports or network reads. The checkpoint owns the RoPE recipe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FACTORS = (1.5, 2.0, 4.0)


def parse_size(value: str) -> int:
    """--context uses binary K/M (case insensitive); plain integers are tokens."""
    match = re.fullmatch(r"([0-9]+)([kKmM]?)", str(value).strip())
    if not match or int(match[1]) < 1:
        raise ValueError("context must be positive tokens or SIZE[k|m], e.g. 512k")
    return int(match[1]) * {"": 1, "k": 1024, "m": 1048576}[match[2].lower()]


def vllm_size(value: str) -> int:
    """Keep vLLM's extra-argument units: lower-case decimal, upper-case binary."""
    value = value.strip()
    if value.lower() == "auto" or value == "-1":
        return -1
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmMgGtT])", value)
    if match:
        number, suffix = match.groups()
        power = "kmgt".index(suffix.lower()) + 1
        return (int(number) * 1024**power if suffix.isupper()
                else int(float(number) * 1000**power))
    return int(value)


def context_args(size: str | int | None, extra: list[str], flags: dict) -> list[str]:
    if size is None:
        return list(extra)
    tokens = parse_size(str(size))
    if "--max-model-len" in flags:
        raise ValueError("give --context or --max-model-len, not both")
    return [*extra, "--max-model-len", str(tokens)]


def factor_for(size: int, native: int) -> float | None:
    if size <= native:
        return None
    for factor in FACTORS:
        if size <= int(native * factor):
            return factor
    raise ValueError(
        f"context {size:,} exceeds the automatic YaRN limit {native * 4:,}; "
        "use an explicit --hf-overrides recipe"
    )


def yarn_override(config: dict, size: int) -> tuple[dict | None, float | None]:
    key = "text_config" if "text_config" in config else None
    text = config[key] if key else config
    native = text.get("max_position_embeddings")
    if not isinstance(native, int) or native < 1:
        raise ValueError("context: config.json needs max_position_embeddings")
    if size <= native:
        return None, None
    params = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if not isinstance(params, dict):
        raise ValueError("context: layered RoPE needs an explicit --hf-overrides")
    if params.get("rope_type", params.get("type", "default")) != "default":
        return None, None  # checkpoint's own scaling always wins
    params = dict(params)
    for name in ("rope_theta", "partial_rotary_factor", "mrope_section",
                 "mrope_interleaved"):
        if name not in params and name in text:
            params[name] = text[name]
    for name in ("rope_theta", "partial_rotary_factor"):
        if name not in params:
            raise ValueError(
                f"context: config.json lacks {name}; give --hf-overrides "
                "with the model's own YaRN recipe"
            )
    factor = factor_for(size, native)
    params.pop("type", None)
    params.update(rope_type="yarn", factor=factor,
                  original_max_position_embeddings=native)
    override = {"max_position_embeddings": int(native * factor),
                "rope_parameters": params}
    return ({key: override} if key else override), factor


# (factor, drafter) reference-host runs of the exact recipe yarn_override generates, max_position_embeddings raised:
# runs at 393,216, 524,288 and 917,504 tokens. The factor-2 run with DFlash2 used the
# rope-only recipe, which is a different override, so generated factor 2 with DFlash2 is not in this table.
TESTED_RECIPES = {
    (1.5, "mtp"): "recipe tested on Flash-Next 4x3090 with mtp",
    (1.5, "dflash2"): "recipe tested on Flash-Next 4x3090 with dflash2",
    (2.0, "mtp"): "recipe tested on Flash-Next 4x3090 with mtp",
    (4.0, "mtp"): "recipe tested with MTP to 917,504 (917K needles 7/8 overall)",
}


def qualification(checkpoint: str, profile: str, spec: str, factor: float) -> str:
    """The label of the generated recipe: tested only on the checkpoint, layout and drafter that ran that exact override."""
    if checkpoint == "flash-next" and profile == "fn-quad-3090" and (factor, spec) in TESTED_RECIPES:
        return TESTED_RECIPES[factor, spec]
    # The generated factor-2 recipe with DFlash2 ships, labelled untested until its qualification run.
    if checkpoint == "flash-next" and profile == "fn-quad-3090" and (factor, spec) == (2.0, "dflash2"):
        return "\u26ab untested pending gate: generated factor-2 recipe with DFlash2 (only the rope-only recipe ran)"
    return "\u26ab untested model/layout/drafter x YaRN factor"


def pool_note(plan, size: int, flags: dict) -> str:
    """Conservative pool estimate at the resolved pin, never counting host RAM."""
    srv = plan.profile["serve"]
    spec = plan.caller_env.get("SPEC") or "mtp"
    dfl = srv.get("dflash2") if spec == "dflash2" else None
    if spec == "dflash2" and not dfl:  # the row's pool is the MTP one: no DFlash2 pool is known for this layout
        raise ValueError(f"context: {plan.profile['id']} has no measured DFlash2 pool")
    cap = dfl or plan.profile.get("capacity", {})
    reference = (dfl or srv).get("kv_cache_memory")
    pool = cap.get("pool_tokens")
    pin = flags.get("--kv-cache-memory-bytes")
    if not pool or not reference or pin is None or pin == "auto":
        raise ValueError("context: cannot resolve a pinned KV pool for this layout")
    # Geometry changes invalidate the registry capacity. Ask for a matching
    # profile rather than claiming that the old pool fits a new geometry.
    for flag, expected in (
        ("--tensor-parallel-size", plan.profile.get("layout", {}).get("tp", srv.get("tensor_parallel", 1))),
        ("--pipeline-parallel-size", plan.profile.get("layout", {}).get("pp", 1)),
        ("--kv-cache-dtype", srv.get("kv_cache_dtype")),
    ):
        actual = flags.get(flag)
        if flag == "--kv-cache-dtype":
            actual = {"fp8": "fp8_e4m3", "bf16": "bfloat16"}.get(actual, actual)
        if expected is not None and actual is not None and str(expected) != actual:
            raise ValueError(f"context: {flag} changes the pool geometry; use a matching profile")
    partition = plan.caller_env.get("PP_PARTITION")
    expected_partition = (dfl or plan.profile.get("layout", {})).get("partition")
    if partition and expected_partition and partition != ",".join(map(str, expected_partition)):
        raise ValueError("context: PP_PARTITION changes the pool geometry; use a matching profile")
    pin = vllm_size(pin)
    # Round down to whole reference blocks where the registry records them.
    blocks = cap.get("kv_blocks")
    if dfl:
        # The DFlash2 pin is a round byte figure, not whole blocks: the engine keeps pin // block bytes blocks, and
        # its pool follows from the block count (the registry's measured formula, prefill runs on).
        import registry as R
        pool = R._dflash2_pool(pin // R.DFLASH2_KV_BLOCK_BYTES)
    elif blocks:
        resolved_blocks = pin * blocks // reference
        pool = pool * resolved_blocks // blocks
    else:
        pool = pool * pin // reference
    if size > pool:
        raise ValueError(
            f"context {size:,} does not fit the resolved KV pool (~{pool:,} "
            f"tokens, {spec}); lower --context/--max-model-len. "
            "KV RAM cache adds no per-request context."
        )
    sessions = min(int(flags["--max-num-seqs"]), pool // size)
    return (f"context: {size:,} tokens; resolved KV pool ~{pool:,}; "
            f"estimated resident full-context sessions {sessions} "
            f"({pool / size:.2f}x pool/context), scheduler cap "
            f"{flags['--max-num-seqs']}. Boot reports actual capacity.")


def apply(plan, flags: dict, extra_flags: dict, default_length: int) -> None:
    requested = ("--max-model-len" in extra_flags
                 or bool(plan.caller_env.get("MAXLEN")))
    if not requested:
        return
    size = vllm_size(flags["--max-model-len"])
    if size == -1:  # retain vLLM's auto-fit path
        return
    if size < 1:
        raise ValueError("context must be positive")
    if any(k in extra_flags for k in ("--block-size", "--num-gpu-blocks-override")):
        raise ValueError("context: custom block geometry has no resolved pool estimate")
    plan.notes.append(pool_note(plan, size, flags))
    if any(k == "--hf-overrides" or k.startswith("--hf-overrides.")
           or k == "--rope-scaling" or k.startswith("--rope-scaling.")
           for k in flags):
        plan.notes.append("context: explicit override wins; no auto-YaRN added")
        return
    path = Path(plan.ckpt) / "config.json"
    if not path.is_file():
        # Default/smaller dry plans need no invented recipe. Extensions require
        # the real checkpoint (setup checks the pool before downloading it).
        if size <= default_length:
            return
        raise ValueError(f"context: {path} is required to build the model's YaRN recipe")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"context: cannot read {path}: {e}") from e
    if plan.model not in ("flash-next", "27b", "ornith"):
        text = config.get("text_config", config.get("llm_config", config))
        if size > text.get("max_position_embeddings", 0):
            raise ValueError("context extension for this model needs explicit --hf-overrides")
        return
    override, factor = yarn_override(config, size)
    if override is None:
        return
    plan.args += ["--hf-overrides", json.dumps(override, separators=(",", ":"))]
    plan.env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"  # tokenizer may cap native
    spec = plan.caller_env.get("SPEC") or "mtp"
    plan.notes += [
        f"context: auto-YaRN factor {factor:g}; "
        + qualification(plan.checkpoint, plan.profile["id"], spec, factor),
        "context: drafter keeps native RoPE; acceptance may drop past native context",
        "context: FP8 KV scales were not calibrated under YaRN",
    ]

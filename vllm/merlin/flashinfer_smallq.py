# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Plan-local FlashInfer small-q selection; no CUDA or vLLM imports."""


def select_smallq_mode(
    *,
    enabled: bool,
    q_len: int,
    query_lens: list[int],
    prefill_flags: list[bool] | None,
    has_prefill: bool,
    cta32_allowed: bool = True,
) -> int:
    """-1 preserves the legacy process switch; 0/1 request stock/cta32.

    Without the cta32 process switch there is nothing to choose between, so the
    legacy path (-1) keeps its ordinary graph wrappers and FULL replay.
    """
    if not enabled or not cta32_allowed or q_len not in (4, 5):
        return -1
    if (
        has_prefill
        or not query_lens
        or prefill_flags is None
        or len(prefill_flags) != len(query_lens)
        or any(q not in (0, q_len) for q in query_lens)
        or not any(query_lens)
        or any(flag and q > 0 for flag, q in zip(prefill_flags, query_lens))
    ):
        return 0
    return 1


def decode_wrapper_key(batch_size: int, q_len: int, mode: int) -> tuple:
    # Preserve existing single-token draft lookups when using the legacy switch.
    return (batch_size, q_len) if mode == -1 else (batch_size, q_len, mode)


def plan_identity(metadata) -> tuple:
    """Include each layer and microbatch; legacy metadata keeps existing keys."""
    groups = metadata if isinstance(metadata, list) else [metadata]
    return tuple(
        (index, name, mode)
        for index, group in enumerate(groups)
        if isinstance(group, dict)
        for name, value in sorted(group.items())
        if (mode := getattr(value, "flashinfer_smallq_mode", -1)) != -1
    )


def scheduled_prefill_flags(input_batch, num_reqs: int) -> list[bool]:
    """Preserve prompt work before hybrid execution routing clears its flags."""
    flags = input_batch.is_prefilling_np[: input_batch.num_reqs].tolist()
    return flags + [False] * (num_reqs - len(flags))


_EVENTS: dict[str, int] = {}


def note_event(kind: str, log) -> int:
    """Count one opt-in tile event; log the first and every 1000th (for test runs)."""
    count = _EVENTS.get(kind, 0) + 1
    _EVENTS[kind] = count
    if count == 1 or count % 1000 == 0:
        log("[flashinfer-workload-tile] %s count=%d", kind, count)
    return count

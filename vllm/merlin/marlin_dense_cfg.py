# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Exact-shape W8A16 tile overrides; unset/blank means OFF.

Syntax: sm=86,sms=82,m=1,n=8192,k=2560,g=64x128,ack=bitwise
Separate exact rules with ';'. No wildcards, ranges, or shipped tuning entries.
g is THREAD_K x THREAD_N; threads = THREAD_K * THREAD_N / 64.
Only BF16, channelwise uint8b128 GDN shapes use this table. A rule also
matches the device SM count because the grid affects the K reduction.

ack=bitwise records a completed bitwise-equality qualification; it does
not prove equality. Changing a tile can change the reduction tree. Restart
after changing the table: captured graphs retain their launch parameters.
This module uses only the standard library so CPU tests need no CUDA.
"""

from functools import lru_cache
from typing import NamedTuple

ENV_NAME = "MERLIN_MARLIN_DENSE_CFG"
GDN_NK = ((8192, 2560), (4096, 2560), (2560, 3072), (2560, 1536))
SMALL = ((128, 128), (64, 128), (128, 64))
LARGE = ((64, 256), (64, 128), (128, 64))


class Rule(NamedTuple):
    sm: int
    sms: int
    m: int
    n: int
    k: int
    tk: int
    tn: int


def split_rows(m: int, n: int) -> tuple[int, ...]:
    """Distinct sublaunch sizes from marlin_mm (not padded graph sizes)."""
    limit = 8192 if n <= 4096 else 1024
    parts = [limit] if m >= limit else []
    tail = m % limit
    if tail >= 64:
        parts.append(tail // 64 * 64)
    if tail % 64:
        parts.append(tail % 64)
    return tuple(parts)


def compiled_configs(m: int, n: int) -> tuple[tuple[int, int], ...]:
    parts = split_rows(m, n)
    return tuple(
        cfg for cfg in SMALL + (LARGE[0],)
        if all(cfg in (SMALL if p <= 16 else LARGE) for p in parts)
    )


def required_smem(m: int, tk: int, tn: int) -> int:
    """Native get_kernel_cache_size, W8/BF16/channel scales, four stages."""
    bm = min((m + 15) // 16, 4) * 16
    b = 4 * tk * tn
    red = bm * (tn + 8) * 2
    tmp = max(b, red, min(b, red) + tn * 2)
    return tmp + 8 * bm * tk + 8 * tn


@lru_cache(maxsize=32)
def parse_rules(raw: str | None) -> tuple[Rule, ...]:
    if not raw or not raw.strip():
        return ()
    rules = []
    seen = set()
    for text in raw.split(";"):
        fields = {}
        for item in text.split(","):
            pair = item.strip().split("=")
            if len(pair) != 2 or pair[0].strip() in fields:
                raise ValueError(f"{ENV_NAME}: malformed or duplicate field")
            fields[pair[0].strip()] = pair[1].strip()
        if set(fields) != {"sm", "sms", "m", "n", "k", "g", "ack"}:
            raise ValueError(f"{ENV_NAME}: require sm,sms,m,n,k,g,ack")
        if fields["ack"] != "bitwise":
            raise ValueError(f"{ENV_NAME}: require ack=bitwise")
        numbers = [fields[key] for key in ("sm", "sms", "m", "n", "k")]
        numbers += fields["g"].split("x")
        if len(numbers) != 7 or any(
            not s.isascii() or not s.isdecimal() or len(s) > 10
            for s in numbers
        ):
            raise ValueError(f"{ENV_NAME}: require positive decimal integers")
        r = Rule(*map(int, numbers))
        if any(v <= 0 or v > 2147483647 for v in r):
            raise ValueError(f"{ENV_NAME}: integer outside positive int32 range")
        if r.sm not in (80, 86, 89) or (r.n, r.k) not in GDN_NK:
            raise ValueError(f"{ENV_NAME}: unsupported architecture or GDN shape")
        if (r.tk, r.tn) not in compiled_configs(r.m, r.n):
            raise ValueError(f"{ENV_NAME}: tile not compiled for every sublaunch")
        key = r[:5]
        if key in seen:
            raise ValueError(f"{ENV_NAME}: duplicate shape")
        seen.add(key)
        rules.append(r)
    return tuple(rules)


def select_override(
    rules: tuple[Rule, ...], *, sm: int, sms: int, m: int, n: int, k: int,
    max_shared_mem: int,
) -> tuple[int, int] | None:
    for r in rules:
        if r[:5] != (sm, sms, m, n, k):
            continue
        if any(
            required_smem(p, r.tk, r.tn) > max_shared_mem
            for p in split_rows(m, n)
        ):
            raise ValueError(f"{ENV_NAME}: tile exceeds device shared memory")
        return r.tk, r.tn
    return None


@lru_cache(maxsize=1)
def env_rules() -> tuple[Rule, ...]:
    """The MERLIN_MARLIN_DENSE_CFG table, read and parsed once per process (at
    the first dense Marlin GEMM), not per call. Empty when unset or blank. A
    malformed table is not cached, so it raises on every call."""
    import vllm.envs as envs

    return parse_rules(envs.MERLIN_MARLIN_DENSE_CFG)


@lru_cache(maxsize=None)
def device_limits(index: int | None) -> tuple[int, int, int]:
    """(sm, SM count, opt-in shared memory per block) of one CUDA device,
    queried once per device."""
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    props = torch.cuda.get_device_properties(index)
    return (
        major * 10 + minor,
        props.multi_processor_count,
        props.shared_memory_per_block_optin,
    )

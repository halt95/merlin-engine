# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier (MERLIN_KV_HOST_TIER_GIB).

One knob, default 0 (off). A positive value is GiB of host memory per rank
(TP x PP); the engine then configures MerlinKVHostTierConnector itself: the
upstream SimpleCPUOffloadConnector in ``kv_both`` mode, run write-back (GPU L1
+ host L2, one cache: vllm/merlin/kv_host_tier_writeback.py). There is no
mode knob. Off leaves ``kv_transfer_config`` exactly as given, so a
connector-less boot is unchanged.

The engine tells the tier apart from a user-supplied ``--kv-transfer-config``
by matching the exact config this module generates (``is_kv_host_tier``). Only
that config gets the four exemptions below; any other connector keeps the
upstream rules:

- RecoverSSM: ``--use-replayssm`` refuses every KV connector upstream. The tier
  is allowed under Flash-Next (Qwen4Exp) GDN/PLE, dense Qwen3.5 GDN (the
  27B lanes) and Qwen3.5 MoE GDN (Ornith 1.5) RecoverSSM in align mode
  on Model Runner V2 (``recoverssm_allows``). Its stores copy only blocks the
  scheduler hashed, after a compute-done event; boundary hand-offs are held
  until every rank acknowledges the actual writer's completion; on
  non-last PP ranks the runner frees a step's finished/preempted rows before the deferred commit
  (``free_before_deferred_commit``), so a stale commit can never write into a
  block the scheduler has reallocated to a host load.
- Side-cache layout: kept (``keeps_side_cache_layout``). The slot-packed QSA
  ring group is not prefix-cacheable, so the tier never stores, finds or loads
  it; the worker copies whole pool blocks by the other groups' ids and skips
  slot-packed tensor views; the folded PLE group offloads like any Mamba
  group. Turning the layout off would cost pool capacity.
- Prefix-cache pins: kept (``keeps_prefix_pins``). The tier holds GPU blocks
  only by additive refs (touch), which the pins' yield accounting excludes,
  and its async loads leave the request WAITING_FOR_REMOTE_KVS, never pinned.
- Config hash: the tier hashes like no connector (``hashes_like_no_connector``).
  Its per-layer hooks are no-ops and KVTransferConfig.compute_hash has no
  factors, so ON and OFF share one compile cache.

Recognition is by exact shape, not provenance (pydantic copies carry no
private marker): an identical hand-written config with the knob on is
functionally the tier and is treated as it.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

KNOB = "MERLIN_KV_HOST_TIER_GIB"
TIER_KEY = "merlin_kv_host_tier"
# The tier contract the generated config carries under TIER_KEY. A config
# whose marker is True (an eager tier) is not this tier and is refused.
TIER_CONTRACT = "write-back"
CONNECTOR = "MerlinKVHostTierConnector"

# Architectures whose RecoverSSM path the tier is qualified for: Flash-Next
# (Qwen4Exp) GDN/PLE, dense Qwen3.5 GDN (Qwen3.8-27B, ThinkingCap) and
# Qwen3.5 MoE GDN (Ornith 1.5: the same GDN layers, RecoverSSM state and
# FA + MTP KV groups as the dense model). The NemotronH Mamba-2 RecoverSSM
# lanes are not.
_QUALIFIED_RECOVERSSM_ARCHS: tuple[str, ...] | None = None


def _qualified_archs() -> tuple[str, ...]:
    global _QUALIFIED_RECOVERSSM_ARCHS
    if _QUALIFIED_RECOVERSSM_ARCHS is None:
        from vllm import flashnext_family
        from vllm.config.vllm import (
            _DENSE_QWEN35_RECOVERSSM_ARCHS,
            _QWEN35_MOE_RECOVERSSM_ARCHS,
        )

        _QUALIFIED_RECOVERSSM_ARCHS = (
            *flashnext_family.archs(),
            *_DENSE_QWEN35_RECOVERSSM_ARCHS,
            *_QWEN35_MOE_RECOVERSSM_ARCHS,
        )
    return _QUALIFIED_RECOVERSSM_ARCHS


def tier_gib() -> int:
    import vllm.envs as envs

    return envs.MERLIN_KV_HOST_TIER_GIB


def extra_config(gib: int, world_size: int) -> dict[str, Any]:
    """The kv_connector_extra_config the knob generates."""
    per_rank = gib << 30
    return {
        # Both keys agree, so the connector logs no mismatch warning.
        "cpu_bytes_to_use": per_rank * world_size,
        "cpu_bytes_to_use_per_rank": per_rank,
        TIER_KEY: TIER_CONTRACT,
    }


def is_kv_host_tier(
    kv_transfer_config: "KVTransferConfig | None", world_size: int
) -> bool:
    """True only for the exact config the knob generates, with the knob on."""
    if kv_transfer_config is None:
        return False
    gib = tier_gib()
    return (
        gib > 0
        and kv_transfer_config.kv_connector == CONNECTOR
        and kv_transfer_config.kv_connector_module_path is None
        and kv_transfer_config.kv_role == "kv_both"
        and (kv_transfer_config.kv_connector_extra_config or {})
        == extra_config(gib, world_size)
    )


def is_active(vllm_config: "VllmConfig") -> bool:
    return is_kv_host_tier(
        getattr(vllm_config, "kv_transfer_config", None),
        vllm_config.parallel_config.world_size,
    )


def refuse_model_runner_v1(vllm_config: "VllmConfig") -> None:
    """Refuse a real engine boot whose worker resolves to Model Runner V1 with
    the tier active. Only MRV2 starts the writer tracker;
    MRV1 would acknowledge step zero forever, so no published content could
    become ready for write-back. Called from the GPU worker's init, where the
    runner is actually chosen and before any arena is allocated; config
    construction and scheduler/connector unit tests never pick a runner and
    are not affected."""
    if is_active(vllm_config) and not vllm_config.use_v2_model_runner:
        raise ValueError(
            f"{KNOB} requires Model Runner V2: Model Runner V1 does not "
            "advance the host-tier commit tracker. Set "
            "VLLM_USE_V2_MODEL_RUNNER=1 and remove features that force V1, "
            f"or set {KNOB}=0."
        )


def refuse_prefix_replay(kv_cache_config) -> None:
    """Refuse the tier on a KV layout with bounded prefix replay (DeepSeek-V4.1
    SWA): the host restore (``PrefixHitPlan``) does not model the replay
    window, and no suffix restore serves that lane.
    Called where the scheduler-side manager is built and again where the
    worker allocates the arena, so it fails before any host memory is
    allocated."""
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or ()
    if any(g.kv_cache_spec.prefix_replay_tokens > 0 for g in groups):
        raise ValueError(
            f"{KNOB} does not support bounded prefix replay (DeepSeek-V4.1 "
            f"sliding-window layouts): set {KNOB}=0 for this model."
        )


def post_init(vllm_config: "VllmConfig") -> None:
    """Knob > 0: build the tier's KVTransferConfig on ``vllm_config``.

    Refuses rather than merges when another connector or kv offloading is
    configured. The exact config it generated itself is accepted as is, so
    re-validation through ``utils.replace`` (the MTP draft config,
    with_hf_config, load/attention overrides) is idempotent. Configs without
    a model (the PLE offload process's process-group setup, benchmarks) serve
    nothing and get no tier.
    """
    from vllm.merlin.envs import kv_host_tier_fail_loads

    kv_host_tier_fail_loads()  # validate the debug knob even with the tier off
    gib = tier_gib()
    check_gate_envs(gib)
    if gib <= 0 or vllm_config.model_config is None:
        return
    if vllm_config.cache_config.enable_prefix_caching is False:
        # Checked before the re-validation return too, so a replaced config
        # cannot carry the tier with prefix caching off.
        raise ValueError(
            f"{KNOB} needs --enable-prefix-caching: the host tier stores and "
            "finds blocks by their prefix-cache hashes"
        )
    world_size = vllm_config.parallel_config.world_size
    if is_kv_host_tier(vllm_config.kv_transfer_config, world_size):
        return
    if vllm_config.kv_transfer_config is not None:
        raise ValueError(
            f"{KNOB} is set but a KV connector is already configured "
            f"({vllm_config.kv_transfer_config.kv_connector!r}, from "
            "--kv-transfer-config or --kv-offloading-size); use one or the other"
        )
    check_knob_fits(gib, vllm_config.parallel_config)
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.logger import init_logger

    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector=CONNECTOR,
        kv_role="kv_both",
        kv_connector_extra_config=extra_config(gib, world_size),
        # A failed restore copy is reported per request (failed_recving, not
        # per block: block ids are not unique across the hybrid groups); the
        # scheduler then recomputes the request instead of failing it.
        kv_load_failure_policy="recompute",
    )
    init_logger(__name__).info(
        "%s=%d: host-RAM KV tier on, %d GiB per rank x %d ranks (%s, write-back)",
        KNOB,
        gib,
        gib,
        world_size,
        CONNECTOR,
    )


def check_gate_envs(gib: int) -> None:
    """The write-back gate's internal envs (copy credits, escape steps) are
    for the tier only: refuse them with the tier off, and parse them now so a
    bad value fails at boot."""
    import os

    from vllm.merlin import kv_host_tier_writeback as wb

    set_names = [name for name in wb.GATE_ENVS if name in os.environ]
    if set_names and gib <= 0:
        raise ValueError(f"{', '.join(set_names)} require {KNOB} > 0")
    if set_names:
        wb.copy_credits()
        wb.escape_steps()
        wb.n_restore()
        wb.max_assembling()
        # The demotion direction's own row and byte credits.
        wb.copy_credits_d()
        wb.copy_bytes_d()
        wb.max_pending_store_blocks()
        wb.fallback_boundary()
        wb.shared_row_lone()
        wb.evict_duplicates_first()
        wb.restore_start_any_step()
        wb.restore_credits_per_slot()
        wb.restore_seat_hold()


# --- Host memory bounds ----------------------------------------------------
# Warn when the tier would take more than this share of the host memory that
# is available before the engine loads (weights staging, the PLE table and the
# process itself need the rest).
WARN_SHARE_OF_AVAILABLE = 0.75


def _cgroup_reclaimable_bytes() -> int:
    """Page cache charged to the cgroup (inactive and active file lists):
    cgroup usage counts it, but the kernel reclaims it before an allocation
    fails (model files read at load sit there; shards several ranks read are
    promoted to the active list). Shmem/tmpfs is on neither list, so it stays
    counted as used."""
    for path, keys in (
        ("/sys/fs/cgroup/memory.stat", ("inactive_file", "active_file")),
        (
            "/sys/fs/cgroup/memory/memory.stat",
            ("total_inactive_file", "total_active_file"),
        ),
    ):
        found: dict[str, int] = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    name, _, value = line.partition(" ")
                    if name in keys:
                        found[name] = int(value)
        except (OSError, ValueError):
            continue
        if found:
            return sum(found.values())
    return 0


def _read_text(path: str) -> str | None:
    """A kernel text file, or None when it cannot be read. Cgroup paths are
    bytes, so undecodable bytes are kept (surrogateescape), never raised."""
    try:
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            return f.read()
    except OSError:
        return None


def _nested_cgroup_levels(
    read: Callable[[str], str | None] = _read_text,
) -> list[tuple[int, int | None, int]]:
    """(limit, usage, reclaimable page cache) bytes for each
    memory cgroup level between this process's own cgroup (from
    /proc/self/cgroup) and the namespace root, which the vllm utils read
    themselves. Catches a systemd MemoryMax on a unit or scope (or a nested
    slice) that the root-only reading misses. Levels
    without a finite limit are skipped; empty when /proc/self/cgroup is
    unreadable, names the root, or names a path that is not a cgroup under
    the mount. Same walk as tools/merlin/preflight.cgroup_memory_dirs."""
    out: list[tuple[int, int | None, int]] = []
    sets = (
        ("/sys/fs/cgroup", "memory.max", "memory.current", True,
         ("inactive_file", "active_file")),
        ("/sys/fs/cgroup/memory", "memory.limit_in_bytes",
         "memory.usage_in_bytes", False,
         ("total_inactive_file", "total_active_file")),
    )
    lines = (read("/proc/self/cgroup") or "").splitlines()
    for mount, lim_name, use_name, v2, keys in sets:
        rel = None
        for ln in lines:
            parts = ln.strip().split(":", 2)
            if len(parts) == 3 and (
                (v2 and parts[0] == "0" and parts[1] == "")
                or (not v2 and "memory" in parts[1].split(","))
            ):
                rel = parts[2]
                break
        segs = [x for x in (rel or "").split("/") if x and x not in (".", "..")]
        dirs = [mount + "/" + "/".join(segs[:i]) for i in range(len(segs), 0, -1)]
        if not dirs or (
            read(f"{dirs[0]}/{lim_name}") is None
            and read(f"{dirs[0]}/cgroup.procs") is None
        ):
            continue
        for d in dirs:
            def num(name: str, d: str = d) -> int | None:
                v = (read(f"{d}/{name}") or "").strip()
                return int(v) if v.isdigit() and int(v) < (1 << 60) else None

            lim = num(lim_name)
            if lim is None:
                continue
            reclaim = 0
            for line in (read(f"{d}/memory.stat") or "").splitlines():
                name, _, value = line.partition(" ")
                if name in keys and value.strip().isdigit():
                    reclaim += int(value)
            out.append((lim, num(use_name), reclaim))
    return out


def host_memory() -> tuple[int, int, str]:
    """(total, available, source) host RAM in bytes for this process.

    total = MemTotal, capped by the cgroup limit; available = MemAvailable,
    capped by the cgroup's headroom (limit - usage + inactive page cache).
    ``source`` names the binding bound for messages.
    """
    import psutil

    from vllm.utils.cpu_resource_utils import (
        get_cgroup_memory_limit,
        get_cgroup_memory_usage,
    )

    vm = psutil.virtual_memory()
    total, available, source = int(vm.total), int(vm.available), "MemAvailable"
    limit = get_cgroup_memory_limit()
    if limit is not None and limit < total:
        total = limit
    if limit is not None:
        usage = get_cgroup_memory_usage()
        if usage is not None:
            headroom = max(0, limit - usage + _cgroup_reclaimable_bytes())
            if headroom < available:
                available, source = headroom, "cgroup memory.max headroom"
    # Cgroup levels (systemd MemoryMax on a scope/unit)
    for lim, use, reclaim in _nested_cgroup_levels():
        if lim < total:
            total = lim
        if use is not None:
            headroom = max(0, lim - use + reclaim)
            if headroom < available:
                available, source = headroom, "cgroup memory.max headroom"
    return total, available, source


def _ranks_on_this_host(parallel_config: Any) -> int:
    """Workers on this host that each allocate an arena: the TP x PP ranks of
    one replica placed here, times the data-parallel replicas local to this
    host (each replica is an engine of its own with the same knob)."""
    world_size = int(parallel_config.world_size)
    nnodes = int(getattr(parallel_config, "nnodes", 1) or 1)
    dp_local = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    return max(1, world_size // max(1, nnodes)) * max(1, dp_local)


def _memlock_limit() -> int | None:
    """Bytes one process may mlock: None when unlimited or the process has
    CAP_IPC_LOCK (or the platform has no such limit; the allocation-time lock
    check still applies there)."""
    try:
        import resource
    except ImportError:
        return None
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        return None
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    if (int(line.split()[1], 16) >> 14) & 1:  # CAP_IPC_LOCK
                        return None
                    break
    except (OSError, ValueError, IndexError):
        pass
    return int(soft)


MEMLOCK_HINT = (
    "raise the memlock limit for the server (ulimit -l unlimited; systemd "
    "LimitMEMLOCK=infinity; docker --ulimit memlock=-1; LXC "
    "lxc.prlimit.memlock = unlimited) or set the knob to 0 (tier off)"
)


def check_knob_fits(gib: int, parallel_config: Any) -> None:
    """Refuse a knob value this host cannot hold, before anything loads.

    The knob is GiB per rank; every rank on this host allocates its own
    arena. Refused when the total is above the host's memory (MemTotal or the
    cgroup limit) or above what is available now; a warning when it would
    leave less than a quarter of the available memory to the engine.
    """
    ranks = _ranks_on_this_host(parallel_config)
    need = (gib << 30) * ranks
    total, available, source = host_memory()
    gi = float(1 << 30)
    what = (
        f"{KNOB}={gib} asks for {need / gi:.0f} GiB of host RAM ({gib} GiB per "
        f"rank x {ranks} ranks on this host; the knob is per rank)"
    )
    if need > total:
        raise ValueError(
            f"{what}, more than this host has ({total / gi:.1f} GiB). "
            f"Lower {KNOB} or set it to 0 (tier off)."
        )
    if need > available:
        raise ValueError(
            f"{what}, but only {available / gi:.1f} GiB is available now "
            f"({source}). Lower {KNOB}, free host memory, or set it to 0."
        )
    # Host RAM only: the arena is mlocked so it can never be swapped to disk,
    # and the worker refuses an arena it cannot lock. Refuse here too, before
    # weights load, when the limit is known to be too small.
    limit = _memlock_limit()
    if limit is not None and (gib << 30) > limit:
        raise ValueError(
            f"{KNOB}={gib}: the host tier locks its arena in RAM ({gib} GiB per "
            f"rank) so it can never be swapped, but each process may lock only "
            f"{limit / float(1 << 20):.0f} MiB (RLIMIT_MEMLOCK, no CAP_IPC_LOCK); "
            f"{MEMLOCK_HINT}."
        )
    if need > WARN_SHARE_OF_AVAILABLE * available:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "%s; that is %.0f %% of the %.1f GiB available now (%s), leaving "
            "little for the engine itself (weights staging, PLE table)",
            what,
            100.0 * need / max(1, available),
            available / gi,
            source,
        )


# One restore unit = everything a single restore boundary needs (unified
# tier): the full-attention chain (QSA main and compressed index count as
# full attention), the sliding-window tail, one recurrent state per recurrent
# group. QSA ring groups are not prefix-cacheable and never move, so they are
# not in it. A tier with fewer data rows than one unit at this many tokens is
# refused at boot (SimpleCPUOffloadScheduler); one that holds a unit but not
# the unit plus the null row and assembly allowance is only warned about.
RESTORE_UNIT_TOKENS = 262_144


def restore_unit_blocks(
    cfg: "KVCacheConfig", tokens: int = RESTORE_UNIT_TOKENS,
    *, coordinator: Any = None, block_sizes: Any = None,
) -> int:
    """Rows in the complete restore manifest, including EAGLE proof sources.

    Excludes the host null row and assembly allowance; capacity warnings add
    those separately through restore_capacity_blocks. ``block_sizes``: the
    resolved per-group tokens per row (DCP-scaled), as the manifest takes them.
    """
    from vllm.merlin.kv_restore_dependencies import restore_dependency_manifest

    return sum(
        group.blocks for group in restore_dependency_manifest(
            cfg, tokens, coordinator=coordinator, block_sizes=block_sizes
        )
    )


def min_tier_rows(spec: "KVCacheConfig", block_sizes: Any = None) -> int:
    """Minimum host data rows for one 262K restore unit in this layout.

    ``block_sizes`` are the resolved per-group row
    sizes in tokens. Under DCP a sharded group's row holds ``block_size x DCP``
    tokens (each rank keeps its own shard in the same row id), so the spec's
    block size would count twice the rows at DCP2 (Hy4 Flash: 5,128 instead of
    2,568 for the latent + IndexPool + SWA groups). None: the specs' own sizes
    (exact without DCP, the Flash-Next and 27B lanes)."""
    # This is the unit term of the old three-unit L_max calculation: divide
    # out its two restore slots and one assembly slot.
    return restore_unit_blocks(spec, RESTORE_UNIT_TOKENS, block_sizes=block_sizes)


def host_block_bytes(cfg: "KVCacheConfig") -> int:
    """Bytes of one host arena row on this rank (all tensors; they share one
    backing allocation, so tensor 0's size over num_blocks is the row)."""
    if not cfg.kv_cache_tensors or not cfg.num_blocks:
        return 0
    return cfg.kv_cache_tensors[0].size // cfg.num_blocks


def restore_unit_warning(
    unit_bytes_per_rank: int, tier_bytes_per_rank: int, ranks: int
) -> str | None:
    """The warning text, or None when the tier meets its capacity budget.

    Warn only: a small tier still works (anti-thrash stays on at every size);
    long-prefix hits may fall back to recompute when restore and assembly
    dependencies cannot be retained together.
    """
    if unit_bytes_per_rank <= 0 or tier_bytes_per_rank >= unit_bytes_per_rank:
        return None
    gi = float(1 << 30)
    return (
        f"{KNOB}={tier_bytes_per_rank / gi:.2f} GiB per rank "
        f"({tier_bytes_per_rank * ranks / gi:.1f} GiB over {ranks} ranks) is smaller "
        "than the 262K restore capacity budget "
        f"({unit_bytes_per_rank / gi:.2f} GiB per rank, "
        f"{unit_bytes_per_rank * ranks / gi:.1f} GiB over {ranks} ranks): a "
        "complete dependency set plus the host null row and assembly "
        "allowance may not fit; long-prefix hits can fall back to recompute. "
        f"Raise {KNOB} to at least "
        f"{-(-unit_bytes_per_rank // (1 << 30))} per rank. Not refused; "
        "GPU anti-thrash stays on."
    )


def warn_if_below_restore_unit(
    cfg: "KVCacheConfig", tier_bytes_per_rank: int, ranks: int,
    *, coordinator: Any = None, assembly_blocks: int | None = None,
) -> str | None:
    """Warn below one complete 262K manifest + the null row."""
    from vllm.merlin.kv_restore_dependencies import restore_capacity_blocks

    blocks = restore_unit_blocks(cfg, coordinator=coordinator)
    capacity = restore_capacity_blocks(blocks, assembly_blocks=assembly_blocks)
    msg = restore_unit_warning(
        capacity * host_block_bytes(cfg), tier_bytes_per_rank, ranks
    )
    if msg:
        from vllm.logger import init_logger

        init_logger(__name__).warning("host-RAM KV tier: %s", msg)
    return msg


def check_arena_fits(arena_bytes: int) -> None:
    """One rank, right before it allocates its arena (the engine's own memory
    is resident by then): refuse instead of being OOM-killed mid-boot."""
    _, available, source = host_memory()
    if arena_bytes > available:
        gi = float(1 << 30)
        raise RuntimeError(
            f"host-RAM KV tier: this rank's arena needs {arena_bytes / gi:.2f} "
            f"GiB but only {available / gi:.2f} GiB of host memory is available "
            f"({source}); lower {KNOB} (GiB per rank) or set it to 0"
        )


# One lock file per host (per container: each has its own /tmp and its own
# memory bounds). Held from the arena check until the arena is allocated and
# locked, so the next rank's check sees it as used.
ARENA_LOCK_PATH = "/tmp/merlin-kv-host-tier.lock"
# Seconds a rank waits for that lock before it stops with an error naming the
# file (MERLIN_KV_HOST_TIER__ARENA_LOCK_TIMEOUT_S overrides it). One rank holds
# the lock for its own allocate + mlock only, and the ranks of a boot take it in
# turn, so the wait of the last rank is the sum of every earlier hold. Measured
# holds (log line "SimpleCPUOffloadWorker [CPU]" to "host-RAM KV tier: rank",
# whole seconds, 4 x RTX 3090 and a one-rank 27B): 14 to 18 s at 24 GiB per rank
# (4 ranks in turn: 64 s for the 96 GiB tier), 37 s for the largest single
# arena (52 GiB), about 0.7 s per GiB. 900 s is 14 times the 4-rank queue, 24
# times the largest single hold and about 6 times 4 ranks of 52 GiB at that
# rate, and still a clean error instead of a hang when a holder is stuck.
ARENA_LOCK_TIMEOUT_S = 900
ARENA_LOCK_TIMEOUT_ENV = "MERLIN_KV_HOST_TIER__ARENA_LOCK_TIMEOUT_S"
ARENA_LOCK_POLL_S = 0.2
# A rank that has waited this long for the lock says so once, so a slow queue
# is told apart from a stuck holder before the deadline.
ARENA_LOCK_NOTE_S = 5.0
# How often the lock file may be found deleted or replaced while a rank takes
# the lock (each time the rank opens the path again) before it gives up.
ARENA_LOCK_REOPEN_ROUNDS = 8


def _logger() -> Any:
    from vllm.logger import init_logger

    return init_logger(__name__)


def arena_lock_timeout_s() -> float:
    """The lock wait in seconds: the knob when set, else ARENA_LOCK_TIMEOUT_S."""
    import vllm.envs as envs

    value = envs.MERLIN_KV_HOST_TIER__ARENA_LOCK_TIMEOUT_S
    return float(ARENA_LOCK_TIMEOUT_S if value is None else value)


def _arena_lock_owner(path: str) -> str:
    """Owner and mode of the lock file for an error message (never follows a link)."""
    import os

    try:
        st = os.lstat(path)
    except OSError:
        return "owner unknown"
    return f"owner uid {st.st_uid}, mode {st.st_mode & 0o7777:04o}"


def _widen_new_file(fd: int, path: str) -> None:
    """Make a file this process has just created readable by every account,
    whatever the creator's umask, so servers of other users can open it and
    serialise on it. Only ever called on a file created here by O_EXCL, never
    on one another account made. A refusal does not stop this rank (the lock
    works for the account that made the file); it is logged once and a later
    account that cannot open the file gets the permission message."""
    import os

    try:
        os.fchmod(fd, 0o644)
    except OSError as e:
        _logger().warning(
            "host-RAM KV tier: could not set mode 0644 on the new arena lock "
            "file %s (%s). The lock works for this account, but another "
            "account cannot open it until its owner runs chmod 644 on it.",
            path, e,
        )


def _open_arena_lock(path: str) -> int:
    """Open the shared lock file for flock and return the descriptor.

    The existing file is opened read-only without O_CREAT: a read-only open
    needs no ownership of the file, and the kernel's protected-regular rule
    (fs.protected_regular) refuses an O_CREAT open of another user's file in a
    sticky directory such as /tmp. Only a missing file is created, with
    O_EXCL, and a creator that loses the race opens the file the winner made.
    The file is never unlinked here and never followed through a symlink; what
    is opened must be a regular file. The descriptor is closed on every error
    after the open.
    """
    import errno
    import os
    import stat

    # O_NONBLOCK: a FIFO planted at the path cannot hold the open before the
    # regular-file check below; it has no effect on a regular file or flock.
    base = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0))
    uid = os.getuid() if hasattr(os, "getuid") else "?"
    fd, created = -1, False
    for _ in range(8):
        try:
            fd = os.open(path, base)
            break
        except FileNotFoundError:
            pass
        except PermissionError as e:
            raise RuntimeError(
                f"host-RAM KV tier: cannot open the existing arena lock file "
                f"{path} ({_arena_lock_owner(path)}; this process runs as uid "
                f"{uid}). Every server on this host opens that one file, so it "
                "must be readable by this account: ask its owner to make it "
                "readable (chmod 644)."
            ) from e
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise OSError(
                    errno.ELOOP,
                    f"host-RAM KV tier: the arena lock path {path} is a "
                    f"symbolic link ({_arena_lock_owner(path)}); refusing to "
                    "follow it. Its owner can replace it with a regular file "
                    "when no host-tier server is running.",
                    path,
                ) from e
            raise
        try:
            fd = os.open(path, base | os.O_CREAT | os.O_EXCL, 0o644)
            created = True
            break
        except FileExistsError:
            continue  # another rank created it between the two opens
        except PermissionError as e:
            raise RuntimeError(
                f"host-RAM KV tier: cannot create the arena lock file {path} "
                f"(this process runs as uid {uid}): creating it needs write "
                "permission on its directory. Ask an administrator to make "
                "that directory writable for this account, or to create the "
                "file once with mode 644."
            ) from e
    if fd < 0:
        raise RuntimeError(
            f"host-RAM KV tier: the arena lock file {path} kept appearing and "
            "disappearing while it was being opened; something on this host is "
            "removing it. Find that process and stop it."
        )
    try:
        if created:
            _widen_new_file(fd, path)
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise RuntimeError(
                f"host-RAM KV tier: the arena lock path {path} is not a regular "
                f"file (st_mode {mode:o}, {_arena_lock_owner(path)}); refusing "
                "to lock it. Its owner can replace it with a regular file when "
                "no host-tier server is running."
            )
    except BaseException:
        os.close(fd)
        raise
    return fd


def _fd_names_path(fd: int, path: str) -> bool:
    """True when ``path`` still names the file ``fd`` is open on."""
    import os

    try:
        named = os.lstat(path)
    except FileNotFoundError:
        return False
    held = os.fstat(fd)
    return (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino)


def _acquire_arena_lock(fcntl: Any, path: str, timeout_s: float) -> int:
    """Open the lock file and flock it exclusively, polling until
    ``timeout_s`` has passed; return the locked descriptor.

    The deadline is monotonic and spans every step; a wait interrupted by a
    signal carries on to the same deadline. Winning the flock proves nothing if
    the path was deleted or replaced while this rank waited (a later rank would
    then lock the new file and run beside this one), so after the lock is won
    the descriptor must still be the file the path names; if it is not, it is
    closed and the path opened again, up to ARENA_LOCK_REOPEN_ROUNDS times.
    The descriptor is closed on every error. What this cannot prevent: a file
    removed while a holder is already inside its critical section lets a rank
    that arrives afterwards create a new file and enter beside that holder, so
    nothing may remove the file while a host-tier server is starting.
    """
    import os
    import time

    start = time.monotonic()
    deadline = start + timeout_s
    note_at: float | None = start + ARENA_LOCK_NOTE_S
    for _ in range(ARENA_LOCK_REOPEN_ROUNDS):
        fd = _open_arena_lock(path)
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, InterruptedError):
                    pass
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"host-RAM KV tier: waited {timeout_s:.0f} s for the "
                        f"arena lock {path} ({_arena_lock_owner(path)}); "
                        "another process on this host holds it, either a "
                        "server still allocating its arena or a stuck one. "
                        f"Find the holder with `fuser -v {path}` or "
                        f"`lsof {path}`; a healthy slow host can raise "
                        f"{ARENA_LOCK_TIMEOUT_ENV} (seconds)."
                    )
                if note_at is not None and now >= note_at:
                    note_at = None
                    _logger().info(
                        "host-RAM KV tier: waiting for the host-tier arena lock "
                        "held by another process: %s, deadline %d s",
                        path, timeout_s,
                    )
                time.sleep(min(ARENA_LOCK_POLL_S, deadline - now))
            if _fd_names_path(fd, path):
                return fd
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)  # the path names another file now: lock that one instead
    raise RuntimeError(
        f"host-RAM KV tier: the arena lock file {path} was deleted or replaced "
        f"{ARENA_LOCK_REOPEN_ROUNDS} times while this rank was taking the lock "
        "(the lock file keeps being replaced); something on this host is "
        "removing it. Find that process and stop it."
    )


class arena_reservation:
    """Check + allocate + mlock one rank's arena under a host-wide lock.

    A per-rank check alone races: every rank on the host reads the same
    available memory, all pass, then together allocate more than the host has
    (OOM kill mid-boot). Under the lock each rank checks against what the ranks
    and instances before it left. Off Linux (no fcntl) the lock is
    process-local.
    """

    _local: Any = None

    def __init__(self, arena_bytes: int) -> None:
        self.arena_bytes = arena_bytes
        self._fd: int | None = None
        self._held_local = False
        self._held_since = 0.0

    def __enter__(self) -> "arena_reservation":
        import threading
        import time

        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        if fcntl is not None:
            # One shared path, so servers of every user serialise on it:
            # opened without O_CREAT when it exists (no symlink followed, no
            # inherited descriptor), then locked with a deadline instead of
            # blocking for as long as any holder lives. The setting is read
            # before the file is opened, so a bad value leaves nothing open.
            timeout_s = arena_lock_timeout_s()
            self._fd = _acquire_arena_lock(fcntl, ARENA_LOCK_PATH, timeout_s)
            self._held_since = time.monotonic()
        else:
            if arena_reservation._local is None:
                arena_reservation._local = threading.Lock()
            arena_reservation._local.acquire()
            self._held_local = True
        try:
            check_arena_fits(self.arena_bytes)
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self._release()

    def _release(self) -> None:
        import os
        import time

        held_s = None
        if self._fd is not None:
            held_s = time.monotonic() - self._held_since
            os.close(self._fd)  # closing the only descriptor drops the flock
            self._fd = None
        if self._held_local:
            arena_reservation._local.release()
            self._held_local = False
        if held_s is not None:
            _logger().info(
                "host-RAM KV tier: arena lock held %.1f s for a %.2f GiB arena",
                held_s, self.arena_bytes / float(1 << 30),
            )


def mlock_refused(device: Any, nbytes: int) -> RuntimeError:
    """The worker's error when its arena cannot be locked: fail closed, the
    tier never runs on swappable (possibly disk-backed) memory."""
    limit = _memlock_limit()
    have = "unlimited" if limit is None else f"{limit / float(1 << 20):.0f} MiB"
    return RuntimeError(
        f"host-RAM KV tier: mlock of this rank's arena ({device}, "
        f"{nbytes / float(1 << 30):.2f} GiB tensor) failed (RLIMIT_MEMLOCK "
        f"{have}); the tier refuses to run on swappable memory. {MEMLOCK_HINT}"
    )


def copy_teardown_failed() -> bool:
    """A SimpleCPU copy backend in this process missed its teardown deadline:
    its DMA completion is not established, so a device-wide synchronize could
    wait on it forever (its arenas stay referenced instead)."""
    import sys

    worker = sys.modules.get("vllm.v1.simple_kv_offload.worker")
    return bool(
        worker is not None and getattr(worker, "_FAILED_SHUTDOWN_WORKERS", ())
    )


def log_arena(device: Any, num_blocks: int, bytes_per_block: int, locked: bool) -> None:
    """The tier's one startup line per rank: arena size and residency."""
    import torch

    from vllm.logger import init_logger

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    init_logger(__name__).info(
        "host-RAM KV tier: rank %d (%s) arena %.2f GiB = %d blocks x %d B "
        "(%s=%d GiB per rank), arena not registered with CUDA (no GPU page "
        "tables), mlock %s",
        rank,
        device,
        num_blocks * bytes_per_block / float(1 << 30),
        num_blocks,
        bytes_per_block,
        KNOB,
        tier_gib(),
        "ok" if locked else "unavailable",
    )


def recoverssm_allows(vllm_config: "VllmConfig") -> bool:
    """Whether ``--use-replayssm`` may run with the configured connector.

    Only the tier, only RecoverSSM (spec decode on; plain ReplaySSM keeps its
    replay buffer and writes the pool checkpoint back on flush), only align
    mode on Model Runner V2, only the Flash-Next and Qwen3.5 GDN (dense 27B,
    MoE Ornith) architectures.
    """
    model_config = vllm_config.model_config
    return (
        is_active(vllm_config)
        and vllm_config.cache_config.use_recoverssm
        and vllm_config.cache_config.mamba_cache_mode == "align"
        and vllm_config.use_v2_model_runner
        and model_config is not None
        and model_config.architecture in _qualified_archs()
    )


def keeps_side_cache_layout(vllm_config: "VllmConfig") -> bool:
    return is_active(vllm_config)


def keeps_prefix_pins(vllm_config: "VllmConfig") -> bool:
    return is_active(vllm_config)


def hashes_like_no_connector(vllm_config: "VllmConfig") -> bool:
    return is_active(vllm_config)


def free_before_deferred_commit(vllm_config: "VllmConfig") -> bool:
    """Non-last PP ranks: free this step's finished/preempted rows before the
    deferred RecoverSSM commit of step T - pp_size runs.

    Upstream order is commit, then free. A request finished or preempted in
    this scheduler output still has its old generation when the deferred
    commit runs, so the PP freed-row mask lets its commit through; under PP
    that commit writes a block the scheduler may already have handed to a host
    load (the load copy is not ordered against it). Freeing first bumps the
    row's generation and the mask skips it. Without a connector nothing is
    loaded into reused blocks, so upstream's order is kept there.
    """
    return (
        is_active(vllm_config)
        and vllm_config.cache_config.use_recoverssm
        and vllm_config.parallel_config.pipeline_parallel_size > 1
    )


def update_finished_and_deferred(runner: Any, scheduler_output: Any) -> None:
    """GPUModelRunner.execute_model's first two request-state updates.

    Upstream order (kept unless ``runner.free_before_deferred_commit``): the
    deferred postprocess of step T - pp_size (non-last PP ranks), then this
    scheduler output's frees.
    """
    from vllm.merlin import kv_commit_acks

    kv_commit_acks.begin_step(runner, scheduler_output)
    if runner.free_before_deferred_commit:
        runner.finish_requests(scheduler_output)
        runner.update_pp_decode_requests()
    else:
        runner.update_pp_decode_requests()
        runner.finish_requests(scheduler_output)


def take_widest_stage_tensors(
    cfg: "KVCacheConfig", kv_cache_configs: "list[KVCacheConfig]"
) -> None:
    """Give the scheduler config the widest PP stage's ``kv_cache_tensors``.

    The engine calls this only when a KV connector is configured (any
    connector, not only the tier). "Same config except layer names" does not
    hold for the backing tensor sizes under pipeline parallelism: each stage
    packs its own layers, so bytes per block differ by stage (Flash-Next
    TP2xPP2: stage 1 is 12.5 % wider). SimpleCPUOffloadScheduler derives its
    host block count from these tensors; sized from stage 0 it hands out block
    ids past the wider stage's arena (out-of-bounds host copy, worker SIGSEGV
    once the host pool fills). Every config has the
    same num_blocks, so tensor 0's size orders bytes per block.
    """
    import copy

    widest = max(
        kv_cache_configs,
        key=lambda c: c.kv_cache_tensors[0].size if c.kv_cache_tensors else 0,
    )
    if widest is not kv_cache_configs[0] and widest.kv_cache_tensors:
        cfg.kv_cache_tensors = copy.deepcopy(widest.kv_cache_tensors)


def set_worker_block_count(
    vllm_config: "VllmConfig",
    scheduler_config: "KVCacheConfig",
    worker_configs: "list[KVCacheConfig]",
) -> None:
    """Resolve the scheduler's host pool once, before sending worker configs.

    The scheduler config already carries the widest PP stage's allocation
    (see the function above). Workers skip packed ring views and may have
    narrower stages: deriving a fresh count from their copy regions wastes
    RAM on blocks the scheduler never addresses. Carry its count instead.
    Other connectors and tier-OFF configs are left untouched.
    """
    if not is_active(vllm_config) or not scheduler_config.kv_cache_tensors:
        return

    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    num_blocks = SimpleCPUOffloadScheduler._derive_cpu_config(
        scheduler_config, tier_gib() << 30
    ).num_blocks
    for cfg in (scheduler_config, *worker_configs):
        cfg.merlin_kv_host_tier_num_blocks = num_blocks


def worker_block_count(cfg: "KVCacheConfig", capacity_blocks: int) -> int:
    """Use the engine-resolved count; fail before allocating on a bad handoff."""
    num_blocks = cfg.merlin_kv_host_tier_num_blocks
    if num_blocks is None:
        raise RuntimeError(
            "host-RAM KV tier: scheduler block count was not resolved before "
            "worker initialization"
        )
    if not 1 <= num_blocks <= capacity_blocks:
        raise RuntimeError(
            f"host-RAM KV tier: scheduler block count {num_blocks} is outside "
            f"this worker's capacity (1..{capacity_blocks} blocks); "
            "host pool sizing must use the widest PP stage"
        )
    return num_blocks

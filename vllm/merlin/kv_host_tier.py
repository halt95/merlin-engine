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
        # The demotion direction's own row and byte credits.
        wb.copy_credits_d()
        wb.copy_bytes_d()
        wb.max_pending_store_blocks()
        wb.fallback_boundary()


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
    *, coordinator: Any = None,
) -> int:
    """Rows in the complete restore manifest, including EAGLE proof sources.

    Excludes the host null row and assembly allowance; capacity warnings add
    those separately through restore_capacity_blocks.
    """
    from vllm.merlin.kv_restore_dependencies import restore_dependency_manifest

    return sum(
        group.blocks for group in restore_dependency_manifest(
            cfg, tokens, coordinator=coordinator
        )
    )


def min_tier_rows(spec: "KVCacheConfig") -> int:
    """Minimum host data rows for one 262K restore unit in this layout."""
    # This is the unit term of the old three-unit L_max calculation: divide
    # out its two restore slots and one assembly slot.
    return restore_unit_blocks(spec, RESTORE_UNIT_TOKENS)


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

    def __enter__(self) -> "arena_reservation":
        import os
        import threading

        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        if fcntl is not None:
            # O_NOFOLLOW: a symlink planted at the shared path is refused,
            # not followed; O_CLOEXEC: no child inherits the lock fd. The
            # creation mode is left to the user's umask so servers on the
            # same host keep serialising on the one path.
            flags = (os.O_RDONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_CLOEXEC", 0))
            fd = os.open(ARENA_LOCK_PATH, flags, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except BaseException:
                os.close(fd)
                raise
            self._fd = fd
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

        if self._fd is not None:
            os.close(self._fd)  # closing the only descriptor drops the flock
            self._fd = None
        if self._held_local:
            arena_reservation._local.release()
            self._held_local = False


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

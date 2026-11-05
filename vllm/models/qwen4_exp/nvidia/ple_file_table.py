# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""The PLE table read from the checkpoint's own safetensors files.

``--ple-store disk``: the offload process maps the table's files read-only instead
of holding a 47.68 GiB anonymous copy. Rows are gathered from the maps, so a miss
is a page fault. The kernel can reclaim the pages, and two servers on one host share
one copy. Nothing else moves: same offload process, same transport, no CUDA and no
graph change. Rows are byte-identical to the RAM table's, the transport carries them
in the same layout and the GPU decodes them in the same way.

Why it has this shape:

* **One ``index_select`` per file, not per shard.** About 50 of the 128 shards are
  hit by a 64-row decode step, and a Python task per shard costs more than the whole
  budget of a step. A file's rows share one stride, so the row maths is vectorised.
* **Two paths, chosen by the faults of recent steps.** While steps fault little, a step
  runs serially, which beats a thread pool at 64 rows. Faults cost far more in series
  than in the pool, so a step with about ten faults sends the next ones to the pool:
  a thread blocked in a page fault is the only way this Python gets more than QD1.
  A run of steps with few faults brings the serial path back, resident table or not,
  and one burst costs one slow serial step.
* **More tasks than files when cold.** A task per file caps the pool at ten threads.
* **Thread count is not capped by CPU affinity.** These threads wait on the device.
* **Warm-up: read through a separate fd, then populate the map.** The map is
  ``MADV_RANDOM``, which makes populate read one page at a time; once the fd read has
  filled the cache, populate does no IO and saves the minor fault that every first
  touch would otherwise take on the serving path.

Failure: a file truncated after boot raises SIGBUS and kills this process. The GPU
worker's watchdog (``gpu_worker._start_ple_offload_watchdog``) notices at once and
fails the engine, so no sentinel is needed. A read error during warm-up exits this
process the same way. A stalled device is bounded by the existing PLE progress
deadline. Every gather re-stats the files it used after copying and before returning
rows: a write updates the times before it changes the bytes, so changed rows are never
published. ctime also moves when a file is renamed over or re-linked; the maps would still
hold the booted bytes, but anything that touches the table's files stops the server.

Linux only; imported only when the store is ``disk``.
"""

from __future__ import annotations

import ctypes
import errno
import json
import math
import mmap
import os
import resource
import struct
import sys
import threading
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

INDEX_NAME = "model.safetensors.index.json"
SHARD_INFIX = ".ngram_embedding.shard_"
MAX_HEADER_BYTES = 100 << 20

#: Below this a quiet gather runs serially; at or above it the pool is faster.
POOL_MIN_ROWS_QUIET = 4096
#: Pool task size while faulting: never smaller than this, so a small step keeps about
#: one task per file.
MIN_TASK_ROWS = 32
#: This many major faults in one step ends the quiet state, on either path (a fault
#: costs far more in series than in the pool; the two paths cost the same near here).
DEMOTE_MAJOR_FAULTS = 10
#: This many steps in a row, each under this many major faults, brings it back.
PROMOTE_STEPS = 8
PROMOTE_MAJOR_FAULTS = 5
#: Sampled residency that counts as warm.
WARM_RESIDENCY = 0.99
RESIDENCY_SAMPLES = 1024
#: Warm only if the cache can hold the table with this much to spare.
WARM_HEADROOM_BYTES = 4 << 30
WARM_READ_BYTES = 16 << 20
WARM_POPULATE_BYTES = 1 << 30
TELEMETRY_INTERVAL_S = 60.0
_MADV_POPULATE_READ = 22
#: Not the storage: no such advice on this kernel, or no memory right now. Warm-up stops; serving goes on.
_NOT_STORAGE_ERRNOS = (errno.EINVAL, errno.ENOSYS, errno.ENOMEM, errno.EAGAIN, errno.EINTR)
#: The kernel's overflow uid: a file this process's root does not really own.
_OVERFLOW_UID = 65534
_PAGE = mmap.PAGESIZE


class PleTableFileError(RuntimeError):
    """The checkpoint's files cannot serve this table."""


@dataclass(frozen=True)
class TablePart:
    """One tensor of a table row: ``shard_N.<name>``, ``width`` values, ``row_bytes`` bytes."""

    name: str
    dtypes: tuple[str, ...]
    width: int
    row_bytes: int


def fp8_parts(row_width: int) -> tuple[TablePart, ...]:
    """The FP8 table: one tensor."""
    return (TablePart("weight", ("F8_E4M3",), row_width, row_width),)


def int6_parts() -> tuple[TablePart, ...]:
    """INT6 g32: codes then scales, the byte order of ``interleave_int6_rows``."""
    if sys.byteorder != "little":
        raise PleTableFileError("the INT6 g32 table needs a little-endian host")
    return (TablePart("codes", ("U8",), 120, 120), TablePart("scales", ("F16",), 5, 10))


def read_header(path: str, fd: int) -> tuple[dict, int]:
    """(the safetensors header, the byte offset of the data section)."""
    raw = os.pread(fd, 8, 0)
    if len(raw) != 8:
        raise PleTableFileError(f"{path}: not a safetensors file (shorter than its header size)")
    size = struct.unpack("<Q", raw)[0]
    if size <= 0 or size > MAX_HEADER_BYTES:
        raise PleTableFileError(f"{path}: not a safetensors file (header size {size})")
    body = os.pread(fd, size, 8)
    if len(body) != size:
        raise PleTableFileError(f"{path}: truncated inside its safetensors header")
    try:
        header = json.loads(body)
    except ValueError as error:
        raise PleTableFileError(f"{path}: unreadable safetensors header") from error
    if not isinstance(header, dict):
        raise PleTableFileError(f"{path}: unreadable safetensors header")
    return header, 8 + size


def _table_weight_map(ckpt_dir: str, marker: str) -> dict[str, str]:
    """{tensor key: file name} for this layer's table tensors."""
    index = os.path.join(ckpt_dir, INDEX_NAME)
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PleTableFileError(f"{index}: unreadable safetensors index") from error
    else:
        weight_map = {}
        for name in sorted(os.listdir(ckpt_dir)):
            if name.endswith(".safetensors"):
                path = os.path.join(ckpt_dir, name)
                fd = os.open(path, os.O_RDONLY)
                try:
                    for key in read_header(path, fd)[0]:
                        weight_map[key] = name
                finally:
                    os.close(fd)
    return {k: v for k, v in weight_map.items() if marker in k}


class _Group:
    """Rows reachable through one view: a file's tensors that share one stride."""

    __slots__ = ("base", "file_index", "view")

    def __init__(self, base: int, file_index: int, view: torch.Tensor) -> None:
        self.base = base
        self.file_index = file_index
        self.view = view


def _pool_thread_init() -> None:
    # A team of OpenMP threads per pool thread would oversubscribe the host; the
    # child inherits OMP_NUM_THREADS = cpus // workers from the executor.
    torch.set_num_threads(1)


class PleFileTable:
    """A PLE table served from the checkpoint's safetensors files."""

    def __init__(
        self,
        ckpt_dir: str,
        layer_idx: int,
        num_rows: int,
        split_parts: int,
        parts: Sequence[TablePart],
        *,
        threads: int = 32,
        warm: bool = True,
    ) -> None:
        if threads < 1:
            raise PleTableFileError(f"PLE disk threads must be at least 1, got {threads}")
        if not os.path.isdir(ckpt_dir):
            raise PleTableFileError(
                f"--ple-store disk reads the table from a local checkpoint folder; {ckpt_dir!r} is not one"
            )
        self.ckpt_dir = ckpt_dir
        self.num_rows = num_rows
        self.parts = tuple(parts)
        self.row_bytes = sum(p.row_bytes for p in self.parts)
        self.threads = threads
        self.warm_requested = warm
        self.warm_state = "pending" if warm else "off"
        # Same arithmetic as Qwen4ExpNGramEmbedding.load_weights.
        self.shard_rows = (num_rows + split_parts - 1) // split_parts
        self.num_shards = math.ceil(num_rows / self.shard_rows)
        # One view per stride class of a file; a row is k consecutive view rows per part.
        self._unit = math.gcd(*(p.row_bytes for p in self.parts))
        self._part_units = tuple(p.row_bytes // self._unit for p in self.parts)
        self._one_unit = self._part_units == (1,)
        self._part_offsets = tuple((n, np.arange(n, dtype=np.int64)) for n in self._part_units)
        self._fds: list[int] = []
        self._maps: list[mmap.mmap] = []
        self._paths: list[str] = []
        self._stats: list[tuple[int, int, int]] = []  # (size, mtime_ns, ctime_ns)
        self._spans: list[tuple[int, int, int]] = []  # (file index, start, end): table bytes
        self._groups: list[_Group] = []
        self._shard_group = np.empty(self.num_shards, dtype=np.int64)
        self._shard_unit0 = np.empty((self.num_shards, len(self.parts)), dtype=np.int64)
        self._quiet = False  # recent steps faulted little: small gathers run serially
        self._quiet_steps = 0
        self._demotions = 0
        self._residency_known = True
        self._times: list[float] = []
        self._rows_gathered = 0
        self._serial_gathers = 0
        self._pool_gathers = 0
        self._next_telemetry = time.monotonic() + TELEMETRY_INTERVAL_S
        self._last_majflt = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        self._majflt_seen = self._last_majflt
        self._warm_thread: threading.Thread | None = None
        self._rng = np.random.default_rng(0)
        self._map_bases: list[int] = []
        try:
            self._open(f"layers.{layer_idx}.")
        except BaseException:
            self.close()
            raise
        self._pool = ThreadPoolExecutor(
            threads, thread_name_prefix="ple-store", initializer=_pool_thread_init
        )
        self._quiet = self._resident()

    # ---------------------------------------------------------------- opening

    def _open(self, layer_marker: str) -> None:
        marker = SHARD_INFIX
        table = {
            key: name
            for key, name in _table_weight_map(self.ckpt_dir, marker).items()
            if layer_marker in key
        }
        wanted = {
            f"{shard}.{part.name}": part
            for shard in range(self.num_shards)
            for part in self.parts
        }
        by_suffix = {key[key.index(marker) + len(marker) :]: key for key in table}
        if len(by_suffix) != len(table):
            raise PleTableFileError(
                f"{self.ckpt_dir}: two table tensors have the same shard name under layer "
                f"marker {layer_marker!r}; serve this checkpoint with --ple-store ram"
            )
        missing = sorted(set(wanted) - set(by_suffix))
        if missing:
            raise PleTableFileError(
                f"{self.ckpt_dir}: the PLE table is incomplete: {len(missing)} of {len(wanted)} "
                f"table tensors are not in the checkpoint (first: shard_{missing[0]})"
            )
        extra = sorted(set(by_suffix) - set(wanted))
        if extra:
            raise PleTableFileError(
                f"{self.ckpt_dir}: table tensor shard_{extra[0]} does not belong to this "
                f"{self.num_shards}-shard table in this format; serve this checkpoint with --ple-store ram"
            )
        by_file: dict[str, list[str]] = {}
        for suffix in wanted:
            by_file.setdefault(table[by_suffix[suffix]], []).append(suffix)
        group_ids: dict[tuple[int, int], int] = {}
        shard_seen: dict[int, int] = {}
        for name in sorted(by_file):
            self._open_file(name, by_file[name], by_suffix, wanted, group_ids, shard_seen)

    def _open_file(
        self,
        name: str,
        suffixes: list[str],
        by_suffix: dict[str, str],
        wanted: dict[str, TablePart],
        group_ids: dict[tuple[int, int], int],
        shard_seen: dict[int, int],
    ) -> None:
        # realpath: an INT6 overlay is links, and the pack may be on another disk.
        path = os.path.realpath(os.path.join(self.ckpt_dir, name))
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as error:
            raise PleTableFileError(f"{path}: cannot open a PLE table file: {error}") from error
        self._fds.append(fd)
        st = os.fstat(fd)
        header, data_start = read_header(path, fd)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        try:
            data_end = max((int(v["data_offsets"][1]) for v in tensors.values()), default=0)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise PleTableFileError(f"{path}: bad safetensors offsets") from error
        if data_start + data_end != st.st_size:
            raise PleTableFileError(
                f"{path}: is {st.st_size} bytes, its header says {data_start + data_end}: "
                "the file is truncated or was changed; verify the checkpoint (merlin verify)"
            )
        file_index = len(self._fds) - 1
        table_map = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        self._maps.append(table_map)
        table_map.madvise(mmap.MADV_RANDOM)
        self._paths.append(path)
        self._stats.append((st.st_size, st.st_mtime_ns, st.st_ctime_ns))
        self._residency_known = self._residency_known and _mincore_is_truthful(path, st)
        lo, hi = st.st_size, 0
        spans: list[tuple[int, int]] = []
        with warnings.catch_warnings():
            # The map is read-only and nothing here writes through a view.
            warnings.filterwarnings("ignore", message=".*not writable.*")
            # A read-only map refuses ctypes' from_buffer; torch gives its address.
            self._map_bases.append(torch.frombuffer(table_map, dtype=torch.uint8).data_ptr())
            for suffix in suffixes:
                part = wanted[suffix]
                shard = int(suffix.split(".", 1)[0])
                rows = max(0, min(self.shard_rows, self.num_rows - shard * self.shard_rows))
                key = by_suffix[suffix]
                info = tensors.get(key)
                if info is None:
                    raise PleTableFileError(f"{path}: the index says it holds {key}; its header does not")
                if info.get("dtype") not in part.dtypes or list(info.get("shape", [])) != [rows, part.width]:
                    raise PleTableFileError(
                        f"{path}: {key} is {info.get('dtype')} {info.get('shape')}, the table needs "
                        f"{part.dtypes[0]} {[rows, part.width]}"
                    )
                begin, end = (int(x) for x in info["data_offsets"])
                if end - begin != rows * part.row_bytes or begin < 0:
                    raise PleTableFileError(f"{path}: {key} has bad data offsets {begin}, {end}")
                begin += data_start
                end += data_start
                spans.append((begin, end))
                lo, hi = min(lo, begin), max(hi, end)
                group_key = (file_index, begin % self._unit)
                group = group_ids.get(group_key)
                if group is None:
                    group = len(self._groups)
                    group_ids[group_key] = group
                    self._groups.append(
                        _Group(begin % self._unit, file_index, self._view(table_map, begin % self._unit, st.st_size))
                    )
                # A row's parts are read together: one group per shard.
                if shard_seen.setdefault(shard, group) != group:
                    raise PleTableFileError(
                        f"{path}: shard_{shard}'s tensors are not in one file on one row stride; "
                        "serve this checkpoint with --ple-store ram"
                    )
                part_index = self.parts.index(part)
                self._shard_group[shard] = group
                self._shard_unit0[shard, part_index] = (begin - self._groups[group].base) // self._unit
        spans.sort()
        for (_, a_end), (b_begin, _) in zip(spans, spans[1:]):
            if b_begin < a_end:
                raise PleTableFileError(f"{path}: two PLE table tensors overlap")
        self._spans.append((file_index, lo, hi))

    def _view(self, table_map: mmap.mmap, base: int, size: int) -> torch.Tensor:
        units = (size - base) // self._unit
        return torch.frombuffer(
            table_map, dtype=torch.uint8, offset=base, count=units * self._unit
        ).reshape(-1, self._unit)

    def table_bytes(self) -> int:
        return self.num_rows * self.row_bytes

    def proof_line(self) -> str:
        return (
            f"PLE store: disk ({len(self._paths)} files, {len(self._groups)} groups, mmap, "
            f"{self.threads} threads, warm {self.warm_state}, "
            f"table {self.table_bytes() / 2**30:.2f} GiB)"
        )

    # ------------------------------------------------------------------ gather

    def _plan(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        """(sort order, the group per sorted row, view rows per row, one span per group)."""
        if ids.size and int(ids.min()) < 0:
            raise IndexError("PLE row id below zero")
        shard = ids // self.shard_rows
        local = ids - shard * self.shard_rows
        group = self._shard_group[shard]
        unit0 = self._shard_unit0[shard]
        if self._one_unit:
            units = unit0[:, 0] + local
        else:
            units = np.concatenate(
                [
                    (unit0[:, index] + local * count)[:, None] + offsets
                    for index, (count, offsets) in enumerate(self._part_offsets)
                ],
                axis=1,
            )
        order = np.argsort(group, kind="stable")
        sorted_group = group[order]
        cuts = np.flatnonzero(sorted_group[1:] != sorted_group[:-1]) + 1
        starts = [0, *cuts.tolist()]
        ends = [*cuts.tolist(), int(ids.size)]
        return order, sorted_group, units, list(zip(starts, ends))

    def _task(
        self,
        out: torch.Tensor,
        order: np.ndarray,
        sorted_group: np.ndarray,
        units: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        rows = order[start:end]
        view = self._groups[int(sorted_group[start])].view
        if self._one_unit:
            selected = torch.index_select(view, 0, torch.from_numpy(units[rows]))
        else:
            selected = torch.index_select(
                view, 0, torch.from_numpy(units[rows].reshape(-1))
            ).reshape(-1, self.row_bytes)
        out.index_copy_(0, torch.from_numpy(rows), selected)

    def _pool_task(self, *args: object) -> None:
        # Inference mode is per thread; the serial path enters it once for the whole step.
        with torch.inference_mode():
            self._task(*args)

    def gather(self, ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Fill U8 ``out`` ``[len(ids), row_bytes]`` with the table rows of ``ids``."""
        began = time.perf_counter()
        ids_np = ids.numpy().astype(np.int64, copy=False)
        order, sorted_group, units, spans = self._plan(ids_np)
        rows = int(ids_np.size)
        if self._quiet and rows < POOL_MIN_ROWS_QUIET:
            with torch.inference_mode():
                for start, end in spans:
                    self._task(out, order, sorted_group, units, start, end)
            self._serial_gathers += 1
        else:
            tasks = spans
            if not self._quiet:
                per_task = max(MIN_TASK_ROWS, rows // (2 * self.threads))
                tasks = [
                    (at, min(at + per_task, end))
                    for start, end in spans
                    for at in range(start, end, per_task)
                ]
            futures = [
                self._pool.submit(self._pool_task, out, order, sorted_group, units, start, end)
                for start, end in tasks
            ]
            for future in futures:
                # A task's exception takes the existing raise -> exit -> watchdog path.
                future.result()
            self._pool_gathers += 1
        self._check_files(sorted_group, spans)
        self._note_faults()
        self._times.append(time.perf_counter() - began)
        self._rows_gathered += rows
        if time.monotonic() >= self._next_telemetry:
            self._telemetry()
        return out

    def _check_files(self, sorted_group: np.ndarray, spans: list[tuple[int, int]]) -> None:
        """After copying, before the rows go anywhere: refuse them if a file they came from changed.

        A RAM table keeps its bytes, and a mapped file does not. ctime catches a write
        whose mtime was put back; a write updates the times before it changes the bytes.
        """
        for start, _ in spans:
            index = self._groups[int(sorted_group[start])].file_index
            st = os.fstat(self._fds[index])
            size, mtime, ctime = self._stats[index]
            if st.st_size != size or st.st_mtime_ns != mtime or st.st_ctime_ns != ctime:
                raise RuntimeError(
                    f"PLE table file {self._paths[index]} changed while serving (size {size} -> "
                    f"{st.st_size}, mtime {mtime} -> {st.st_mtime_ns}, ctime {ctime} -> "
                    f"{st.st_ctime_ns}); refusing to serve rows from it"
                )

    def _note_faults(self) -> None:
        """Pick the next step's path from this step's major faults (the whole process: only gathers take them)."""
        now = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        faults = now - self._majflt_seen
        self._majflt_seen = now
        if self._quiet:
            if faults >= DEMOTE_MAJOR_FAULTS:
                self._quiet = False
                self._quiet_steps = 0
                self._demotions += 1
        elif faults < PROMOTE_MAJOR_FAULTS:
            self._quiet_steps += 1
            if self._quiet_steps >= PROMOTE_STEPS:
                self._quiet = True
        else:
            self._quiet_steps = 0

    # --------------------------------------------------------------- telemetry

    def _page_addresses(self, count: int) -> tuple[list[int], int]:
        """Random page addresses over the table's bytes, and the page count they stand for."""
        spans = []
        total = 0
        for file_index, lo, hi in self._spans:
            first = lo // _PAGE
            pages = (hi - 1) // _PAGE - first + 1
            spans.append((self._map_bases[file_index] + first * _PAGE, pages))
            total += pages
        sizes = np.array([pages for _, pages in spans], dtype=np.int64)
        picks = self._rng.integers(0, total, count)
        edges = np.cumsum(sizes)
        out = []
        for pick in picks.tolist():
            index = int(np.searchsorted(edges, pick, side="right"))
            before = int(edges[index - 1]) if index else 0
            out.append(spans[index][0] + (pick - before) * _PAGE)
        return out, total

    def residency(self, samples: int = RESIDENCY_SAMPLES) -> float | None:
        """Share of sampled table pages in the page cache (mincore: no IO, no faults); None when unknown.

        The kernel reports every page resident to a process that can neither own nor
        write the file, so on such a file a cold table would read as warm.
        """
        if not self._residency_known:
            return None
        libc = _libc()
        vec = ctypes.create_string_buffer(1)
        hits = 0
        addresses, _ = self._page_addresses(samples)
        for address in addresses:
            if libc.mincore(ctypes.c_void_p(address), _PAGE, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            hits += vec.raw[0] & 1
        return hits / samples

    def _resident(self) -> bool:
        residency = self.residency()
        return residency is not None and residency >= WARM_RESIDENCY

    def _telemetry(self) -> None:
        self._next_telemetry = time.monotonic() + TELEMETRY_INTERVAL_S
        majflt = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
        times = sorted(self._times)
        gathers = len(times)
        residency = self.residency()
        if residency is not None and residency >= WARM_RESIDENCY:
            self._quiet = True
        logger.info(
            "PLE store: gathers %d (serial %d, pool %d) rows %d gather ms p50 %.3f p99 %.3f "
            "majflt %d (%.1f/gather) residency %s quiet %d demotions %d",
            gathers,
            self._serial_gathers,
            self._pool_gathers,
            self._rows_gathered,
            1e3 * times[gathers // 2],
            1e3 * times[int(gathers * 0.99)],
            majflt - self._last_majflt,
            (majflt - self._last_majflt) / gathers,
            "unknown" if residency is None else f"{100 * residency:.1f} %",
            int(self._quiet),
            self._demotions,
        )
        self._times = []
        self._rows_gathered = self._serial_gathers = self._pool_gathers = self._demotions = 0
        self._last_majflt = majflt

    # ----------------------------------------------------------------- warm-up

    def warm_room_bytes(self, tier_gib: int, ranks: int) -> tuple[int, str]:
        """What the page cache could hold now, less the KV arena still to come.

        READY comes before the KV arenas are allocated, so MemAvailable still counts
        them, all of them: locked memory on the host may be another server's.
        """
        from vllm.merlin.kv_host_tier import host_memory

        _, available, source = host_memory()
        return available - max(0, tier_gib) * ranks * 2**30 - WARM_HEADROOM_BYTES, source

    def start_warm(self, stop: threading.Event, tier_gib: int = 0, ranks: int = 1) -> None:
        """After READY: warm the table if the page cache can hold all of it."""
        if not self.warm_requested:
            logger.info("PLE store: warm-up off; the table stays cold until requests touch it")
            return
        try:
            room, source = self.warm_room_bytes(tier_gib, ranks)
        except Exception as error:  # noqa: BLE001 - no reading, no warm-up; serving goes on
            self.warm_state = "skipped"
            logger.warning("PLE store: warm-up skipped: no host memory reading (%s)", error)
            return
        if room < self.table_bytes():
            self.warm_state = "skipped"
            logger.info(
                "PLE store: warm-up skipped: the page cache could hold %.1f GiB (%s, less the "
                "KV RAM cache and headroom), the table is %.1f GiB; first reads are slower",
                max(room, 0) / 2**30,
                source,
                self.table_bytes() / 2**30,
            )
            return
        self.warm_state = "running"
        self._warm_thread = threading.Thread(
            target=self._warm, args=(stop,), name="ple-store-warm", daemon=True
        )
        self._warm_thread.start()

    def _warm(self, stop: threading.Event) -> None:
        """A read error here fails this process like a SIGBUS would; the watchdog takes it from there."""
        try:
            began = time.monotonic()
            if not self._warm_read(stop):
                return
            read_done = time.monotonic()
            populated = self._populate(stop)
            if stop.is_set():
                return
            residency = self.residency()
            self._quiet = self._quiet or (residency is not None and residency >= WARM_RESIDENCY)
            self.warm_state = "done"
            logger.info(
                "PLE store: warm-up done in %.1f s (read %.1f s, map %.1f s%s), residency %s",
                time.monotonic() - began,
                read_done - began,
                time.monotonic() - read_done,
                "" if populated else "; populate unavailable, first touches take minor faults",
                "unknown" if residency is None else f"{100 * residency:.1f} %",
            )
        except OSError as error:
            if stop.is_set():
                return
            self.warm_state = "failed"
            if error.errno in _NOT_STORAGE_ERRNOS:
                logger.warning("PLE store: warm-up stopped: %s", error)
                return
            logger.error("PLE table read failed during warm-up: %s; stopping the offload process", error)
            _exit_child()
        except Exception as error:  # noqa: BLE001 - not the storage; warm-up is an optimisation
            self.warm_state = "failed"
            logger.warning("PLE store: warm-up stopped: %s", error)

    def _warm_read(self, stop: threading.Event) -> bool:
        """Fill the page cache through fds of their own; False when stopped. /proc: the same inodes the maps hold."""
        buffer = bytearray(WARM_READ_BYTES)
        for file_index, lo, hi in self._spans:
            fd = os.open(f"/proc/self/fd/{self._fds[file_index]}", os.O_RDONLY)
            try:
                os.posix_fadvise(fd, lo, hi - lo, os.POSIX_FADV_SEQUENTIAL)
                at = lo
                while at < hi:
                    if stop.is_set():
                        return False
                    got = os.preadv(fd, [buffer], at)  # releases the GIL
                    if got <= 0:
                        raise OSError(errno.EIO, f"{self._paths[file_index]}: short read at {at}")
                    at += got
            finally:
                os.close(fd)
        return True

    def _populate(self, stop: threading.Event) -> bool:
        """Map the now-cached table: no IO, and no minor fault per first touch afterwards.

        False when this kernel has no such advice or memory ran out: an optimisation lost.
        Anything else is the storage failing, and raises.
        """
        libc = _libc()  # ctypes releases the GIL; mmap.madvise would not
        for file_index, lo, hi in self._spans:
            base = self._map_bases[file_index]
            start = base + lo // _PAGE * _PAGE
            end = base + hi
            while start < end:
                if stop.is_set():
                    return False
                size = min(WARM_POPULATE_BYTES, end - start)
                if libc.madvise(ctypes.c_void_p(start), size, _MADV_POPULATE_READ) != 0:
                    code = ctypes.get_errno()
                    if code in _NOT_STORAGE_ERRNOS:
                        return False
                    raise OSError(code, f"{self._paths[file_index]}: populate failed")
                start += size
        return True

    def close(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        for group in self._groups:
            group.view = None  # type: ignore[assignment]
        self._groups = []
        for table_map in self._maps:
            try:
                table_map.close()
            except BufferError:
                pass  # a view still holds it; the process is exiting
        self._maps = []
        for fd in self._fds:
            os.close(fd)
        self._fds = []


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    return libc


def _mincore_is_truthful(path: str, st: os.stat_result) -> bool:
    """Whether mincore tells this process the truth about this file: only an owner or a
    writer gets it (the kernel's can_do_mincore); others read every page as resident."""
    euid = os.geteuid()
    return st.st_uid == euid or (euid == 0 and st.st_uid != _OVERFLOW_UID) or os.access(path, os.W_OK)


def _exit_child() -> None:
    os._exit(1)


__all__ = [
    "PleFileTable",
    "PleTableFileError",
    "TablePart",
    "fp8_parts",
    "int6_parts",
]

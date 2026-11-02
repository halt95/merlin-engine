#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Repack the built INT6 g32 PLE table into the Flash-Next family's PLE pack.

The table (128 files ``ple-int6g32-shardNNN.safetensors``, each holding
``<prefix>.shard_N.codes`` U8 [rows,120] and ``<prefix>.shard_N.scales`` F16 [rows,5]; 41,600,199,680 data bytes =
38.74 GiB) becomes ``--files`` pack shards ``model-pleint6-0000I-of-0000M.safetensors`` (shards in order, the same
number per file), the manifest ``ple-int6-pack.json`` (dtype, prefix, shard count, rows, weight_map) and the checksum
list ``ple-int6-pack.sha256`` (commit it as models/flash-next/ple-int6-pack.sha256).

CPU only, standard library only, deterministic: the tensor bytes are copied unchanged in a fixed order behind a
fixed header, so the same table gives byte-identical files and checksums on any host. Streams in 64 MiB blocks (peak
RSS well under 200 MiB) and hashes while writing (one read of the table, one write of ~38.74 GiB). Each file is
written as ``<name>.partial`` and renamed when complete.

DRY RUN IS THE DEFAULT: headers are read and checked, the plan is printed, nothing is written. ``--build`` writes.

``--from-bf16 FILE``: no INT6 table at hand: quantise the BF16 PLE table (one
safetensors file, or a folder of them, holding ``<prefix>.shard_N.weight`` BF16 [rows,160]; base Qwen3.8-Flash-Next's
model-00002-of-00033.safetensors carries all 128) with the INT6 g32 recipe:
symmetric groups of 32, FP16 scale = absmax/31, q = clamp(rint(x / scale), -31, 31), u = q + 32 packed four per three
bytes little endian. One shard in memory at a time (peak RSS ~2 GiB); needs numpy.

Checks (all must pass, else exit 1 before anything is written):
  1. the 128 source files exist, each with exactly the two tensors of its own shard number, U8 [rows,120] and
     F16 [rows,5] with equal rows, and data offsets that tile the file;
  2. with ``--base DIR`` (an FP8 Flash-Next checkpoint): each shard's rows equal the base's ``.shard_N.weight`` rows
     and the base maps exactly 128 shard weights + one weight_scale to its model-plefp8-* files.

    make-ple-int6-pack.py [--build] [--src DIR | --from-bf16 FILE|DIR] [--out DIR] [--base DIR] [--files M]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "merlin"))
import ple_pack as P  # noqa: E402

SRC_NAME = "ple-int6g32-shard{:03d}.safetensors"
BLOCK = 64 << 20
DTYPE_BYTES = {"U8": 1, "F16": 2}
ST_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
                  "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
HEADER_CAP = 100 << 20


class BadFile(ValueError):
    pass


def header(path: Path) -> tuple[int, dict]:
    """(data start, header) of a safetensors file; BadFile when the length field or the JSON does not hold up."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise BadFile(f"{path.name}: shorter than a safetensors length field")
        n = struct.unpack("<Q", head)[0]
        if not 0 < n <= min(size - 8, HEADER_CAP):
            raise BadFile(f"{path.name}: header length {n} outside (0, min(file size - 8, {HEADER_CAP})]")
        raw = f.read(n)
    if len(raw) != n:
        raise BadFile(f"{path.name}: short header read")
    try:
        h = json.loads(raw)
    except ValueError:
        raise BadFile(f"{path.name}: header is not JSON") from None
    if not isinstance(h, dict):
        raise BadFile(f"{path.name}: header is not an object")
    return 8 + n, h


def span_problem(k: str, v: dict, region: int) -> str | None:
    """Why tensor ``k``'s data_offsets are not a valid span of ``region`` data bytes for its dtype and shape."""
    try:
        a, b = (int(x) for x in v["data_offsets"])
        nbytes = ST_DTYPE_BYTES[v["dtype"]]
        for d in v["shape"]:
            nbytes *= int(d)
    except (KeyError, TypeError, ValueError):
        return f"{k}: unreadable dtype / shape / data_offsets"
    if not 0 <= a <= b <= region:
        return f"{k}: data_offsets [{a}, {b}] outside the {region}-byte data region"
    if b - a != nbytes:
        return f"{k}: data_offsets span {b - a} bytes, {v['dtype']}{v['shape']} needs {nbytes}"
    return None


def check_source(src: Path, prefix: str, shards: int) -> tuple[list[dict], list[str]]:
    """One record per shard (file, data start, rows, codes/scales offsets) and the failures."""
    recs, fails = [], []
    for n in range(shards):
        p = src / SRC_NAME.format(n)
        if not p.is_file():
            fails.append(f"missing {p}")
            continue
        try:
            start, h = header(p)
        except BadFile as e:
            fails.append(str(e))
            continue
        keys = sorted(k for k in h if k != "__metadata__")
        want = sorted([f"{prefix}.shard_{n}.codes", f"{prefix}.shard_{n}.scales"])
        if keys != want:
            fails.append(f"{p.name}: tensors {keys}, expected {want}")
            continue
        c, s = h[want[0]], h[want[1]]
        try:  # a header entry without dtype / shape / data_offsets is a failure to report, not a KeyError
            rows = c["shape"][0] if c.get("shape") else -1
            if c["dtype"] != "U8" or c["shape"] != [rows, P.INT6_ROW[0]] or s["dtype"] != "F16" or s["shape"] != [rows, P.INT6_ROW[1]]:
                fails.append(f"{p.name}: codes {c['dtype']}{c['shape']} scales {s['dtype']}{s['shape']}, expected U8[rows,120] F16[rows,5]")
                continue
            spans = sorted((t["data_offsets"][0], t["data_offsets"][1], t["dtype"], t["shape"]) for t in (c, s))
        except (KeyError, TypeError, IndexError, AttributeError):
            fails.append(f"{p.name}: a tensor header lacks dtype, shape or data_offsets")
            continue
        for a, b, dt, shp in spans:
            if b - a != shp[0] * shp[1] * DTYPE_BYTES[dt]:
                fails.append(f"{p.name}: a tensor's byte span does not match its shape")
        if spans[0][0] != 0 or spans[0][1] != spans[1][0] or start + spans[1][1] != p.stat().st_size:
            fails.append(f"{p.name}: the data offsets do not tile the file")
        recs.append({"n": n, "path": p, "start": start, "rows": rows, "codes": c["data_offsets"], "scales": s["data_offsets"]})
    return recs, fails


def check_bf16(src: Path, prefix: str, shards: int) -> tuple[list[dict], list[str]]:
    """One record per shard from the BF16 table: where its [rows,160] weight is, and the INT6 byte spans it becomes."""
    files = sorted(src.glob("*.safetensors")) if src.is_dir() else [src]
    where: dict[int, list[tuple[Path, int, dict, str]]] = {}
    pat = re.compile(r"(?:^|\.)ngram_embedding\.shard_(\d+)\.weight$")  # by name, whatever the checkpoint's own prefix
    every: dict[Path, list[tuple[int, int, str]]] = {}  # every tensor's span in the files that hold a PLE shard
    bad: list[str] = []
    for f in files:  # headers only: other files are ignored; other tensors only bound the PLE spans
        try:
            start, h = header(f)
        except BadFile as e:
            bad.append(str(e))
            continue
        hits = []
        for k, v in h.items():
            m = pat.search(k) if k != "__metadata__" else None
            if m:
                hits.append((int(m.group(1)), (f, start, v, k)))
        if hits:  # every tensor of a PLE file must be a valid span before it can bound the PLE ones
            probs = [x for x in (span_problem(k, v, f.stat().st_size - start) for k, v in h.items() if k != "__metadata__") if x]
            if probs:
                bad += [f"{f.name}: {x}" for x in probs[:3]]
                continue
            every[f] = sorted((int(v["data_offsets"][0]), int(v["data_offsets"][1]), k) for k, v in h.items() if k != "__metadata__")
        for n, rec in hits:
            where.setdefault(n, []).append(rec)
    recs, fails = [], list(bad)
    if bad:
        return recs, fails
    for n in range(shards):
        got = where.get(n, [])
        if len(got) != 1:
            fails.append(f"ngram_embedding.shard_{n}.weight: {len(got)} tensors in {src} (expected exactly 1)")
            continue
        f, start, v, k = got[0]
        rows = v["shape"][0] if v.get("shape") else -1
        if v["dtype"] != "BF16" or v["shape"] != [rows, 160] or rows < 1:
            fails.append(f"{k}: {v['dtype']}{v['shape']}, expected BF16[rows,160]")
            continue
        a, b = v["data_offsets"]
        if not (0 <= a <= b and b - a == rows * 320 and start + b <= f.stat().st_size):
            fails.append(f"{k}: data_offsets {v['data_offsets']} do not hold BF16[{rows},160] inside {f.name}")
            continue
        # against EVERY tensor of the file (PLE or not): a header pointing a shard at another tensor's bytes is refused
        clash = [o for (x, y, o) in every.get(f, []) if o != k and x < y and a < y and x < b]  # an empty span holds no bytes
        if clash:
            fails.append(f"{k}: data_offsets {v['data_offsets']} overlap {clash[0]} in {f.name}")
            continue
        recs.append({"n": n, "path": f, "start": start, "rows": rows, "bf16": v["data_offsets"],
                     "codes": [0, rows * P.INT6_ROW[0]], "scales": [0, rows * P.INT6_ROW[1] * 2]})
    return recs, fails


QMAX, G, CHUNK = 31, 32, 250_000


def quantise_shard(r: dict) -> tuple[bytes, bytes]:
    """(codes U8 [rows,120], scales F16 [rows,5]) of one BF16 shard: quantise and pack, chunk by chunk."""
    import numpy as np

    rows, a = r["rows"], r["bf16"][0]
    codes, scales = np.empty((rows, P.INT6_ROW[0]), np.uint8), np.empty((rows, P.INT6_ROW[1]), "<f2")  # little endian on disk
    with open(r["path"], "rb") as f:
        for r0 in range(0, rows, CHUNK):
            n = min(CHUNK, rows - r0)
            f.seek(r["start"] + a + r0 * 320)
            raw = f.read(n * 320)
            if len(raw) != n * 320:
                raise SystemExit(f"{r['path']}: short read in shard {r['n']} (file changed under the repack?)")
            x = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(n, -1, G)
            s = (np.abs(x).max(axis=2) / QMAX).astype("<f2")
            sf = s.astype(np.float32)
            q = np.clip(np.rint(np.divide(x, sf[..., None], out=np.zeros_like(x), where=sf[..., None] > 0)), -QMAX, QMAX)
            u = (q.astype(np.int16) + 32).astype(np.uint32).reshape(n, -1, 4)
            w = u[..., 0] | (u[..., 1] << 6) | (u[..., 2] << 12) | (u[..., 3] << 18)
            codes[r0:r0 + n] = np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], axis=-1).astype(np.uint8).reshape(n, -1)
            scales[r0:r0 + n] = s
    return codes.tobytes(), scales.tobytes()


def check_base(base: Path, prefix: str, recs: list[dict]) -> list[str]:
    idx = json.loads((base / P.INDEX).read_text(encoding="utf-8"))["weight_map"]
    fp8 = [k for k, f in idx.items() if f.startswith(P.FP8_SHARD_PREFIX)]
    fails = []
    if len(fp8) != len(recs) + 1 or not any(k.endswith("weight_scale") for k in fp8):
        fails.append(f"the base maps {len(fp8)} keys to {P.FP8_SHARD_PREFIX}*, expected {len(recs) + 1}")
    for r in recs:
        k = f"{prefix}.shard_{r['n']}.weight"
        if k not in idx:
            fails.append(f"the base has no {k}")
            continue
        _, h = header(base / idx[k])
        if h[k]["shape"][0] != r["rows"]:
            fails.append(f"shard {r['n']}: {r['rows']} INT6 rows, the base FP8 shard has {h[k]['shape'][0]}")
    return fails


def plan_files(recs: list[dict], files: int) -> list[list[dict]]:
    per = -(-len(recs) // files)
    return [recs[i:i + per] for i in range(0, len(recs), per)]


def file_header(group: list[dict], prefix: str) -> tuple[bytes, list[tuple[dict, str, int, int]]]:
    """The safetensors header (fixed key order, space-padded to 8 bytes) and the copy plan (record, part, src a, b)."""
    meta: dict = {"__metadata__": {"format": "pt", "merlin_ple_pack": P.INT6_DTYPE}}
    copies, off = [], 0
    for r in group:
        for part, (cols, dt) in (("codes", (P.INT6_ROW[0], "U8")), ("scales", (P.INT6_ROW[1], "F16"))):
            a, b = r[part]
            meta[f"{prefix}.shard_{r['n']}.{part}"] = {"dtype": dt, "shape": [r["rows"], cols], "data_offsets": [off, off + b - a]}
            copies.append((r, part, a, b))
            off += b - a
    raw = json.dumps(meta, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw, copies


def write_file(path: Path, head: bytes, copies: list) -> str:
    h = hashlib.sha256()
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as out:
        out.write(head)
        h.update(head)
        made: dict[int, tuple[bytes, bytes]] = {}
        for r, part, a, b in copies:
            if "bf16" in r:  # quantised now: codes then scales of the same shard, one shard held at a time
                if r["n"] not in made:
                    made.clear()
                    made[r["n"]] = quantise_shard(r)
                buf = made[r["n"]][0 if part == "codes" else 1]
                if len(buf) != b - a:
                    raise SystemExit(f"shard {r['n']}: {part} is {len(buf)} bytes, the header says {b - a}")
                out.write(buf)
                h.update(buf)
                continue
            with open(r["path"], "rb") as f:
                f.seek(r["start"] + a)
                left = b - a
                while left:
                    buf = f.read(min(BLOCK, left))
                    if not buf:
                        raise SystemExit(f"{r['path']}: short read")
                    out.write(buf)
                    h.update(buf)
                    left -= len(buf)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="write the pack (default: dry run, nothing written)")
    ap.add_argument("--src", help="the built INT6 g32 table (a folder); this or --from-bf16 is required")
    ap.add_argument("--from-bf16", metavar="PATH", help="quantise this BF16 PLE table instead (a safetensors file or folder)")
    ap.add_argument("--out", help="the pack folder to create (must be missing or empty); required with --build")
    ap.add_argument("--base", help="an FP8 Flash-Next checkpoint to cross-check the rows against (recommended)")
    ap.add_argument("--files", type=int, default=8, help="pack shard files (default 8: 16 table shards, ~4.8 GiB each)")
    ap.add_argument("--prefix", default=P.NGRAM_PREFIX, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.files < 1:
        ap.error("--files takes a whole number >= 1")
    if not (args.src or args.from_bf16):
        ap.error("give --src DIR (the built INT6 g32 table) or --from-bf16 PATH")
    src, prefix = Path(args.from_bf16 or args.src), args.prefix
    recs, fails = (check_bf16 if args.from_bf16 else check_source)(src, prefix, P.NGRAM_SHARDS)
    if args.base and not fails:
        fails += check_base(Path(args.base), prefix, recs)
    groups = plan_files(recs, args.files) if not fails else []
    names = [P.INT6_SHARD_PREFIX + f"{i + 1:05d}-of-{len(groups):05d}.safetensors" for i in range(len(groups))]
    data = sum(r["codes"][1] - r["codes"][0] + r["scales"][1] - r["scales"][0] for r in recs)
    report = {"src": str(src), "shards": len(recs), "data_bytes": data, "data_gib": round(data / (1 << 30), 2),
              "files": {n: [g[0]["n"], g[-1]["n"]] for n, g in zip(names, groups)}, "fails": fails, "dry_run": not args.build}
    if fails or not args.build:
        print(json.dumps(report, indent=1))
        return 1 if fails else 0
    if not args.out:
        print("make-ple-int6-pack: --build needs --out DIR", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"make-ple-int6-pack: {out} is not empty; nothing written", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    sums, weight_map = {}, {}
    for name, group in zip(names, groups):
        head, copies = file_header(group, prefix)
        print(f"writing {name}: table shards {group[0]['n']}..{group[-1]['n']}", flush=True)
        sums[name] = write_file(out / name, head, copies)
        for r in group:
            weight_map[f"{prefix}.shard_{r['n']}.codes"] = weight_map[f"{prefix}.shard_{r['n']}.scales"] = name
    manifest = {"format": P.PACK_FORMAT, "ple_embedding_dtype": P.INT6_DTYPE, "prefix": prefix, "shards": len(recs),
                "rows": [r["rows"] for r in recs], "replaces": P.FP8_SHARD_PREFIX + "*",
                "source": ("quantised by make-ple-int6-pack.py --from-bf16 with the INT6 g32 "
                           "group-scale recipe" if args.from_bf16 else "INT6 g32 table") + " (from the reference recipe's BF16 PLE table)",
                "weight_map": dict(sorted(weight_map.items()))}
    (out / P.PACK_MANIFEST).write_bytes((json.dumps(manifest, indent=1) + "\n").encode())
    sums[P.PACK_MANIFEST] = P.sha256_file(out / P.PACK_MANIFEST)
    lines = [f"# SHA-256 of the {len(sums)} files of the Flash-Next family INT6 g32 PLE pack ({data} table bytes);", "# built by models/flash-next/make-ple-int6-pack.py; run 'sha256sum -c' inside the pack folder."]
    lines += [f"{h}  {n}" for n, h in sorted(sums.items())]
    (out / "ple-int6-pack.sha256").write_bytes(("\n".join(lines) + "\n").encode())
    total = sum((out / n).stat().st_size for n in sums)
    report.update(out=str(out), written_bytes=total, list=str(out / "ple-int6-pack.sha256"))
    print(json.dumps(report, indent=1))
    print(f"commit {out / 'ple-int6-pack.sha256'} as {P.PACK_LIST} and set ple_packs.int6 files/bytes = "
          f"{len(sums)}/{total} in release/download-sizes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Build a static cold-expert profile from an fn2_probe routing capture.

Standard library only (no numpy, no torch). Reads the probe's
``seg-*-rank*-pid*.npz`` segments (the routing-capture probe format; any layout:
PP stages are merged by each rank's recorded layer slots), and writes the
``merlin-cold-experts/1`` JSON that ``MERLIN_COLD_EXPERTS`` points at
(format: vllm/merlin/cold_experts_plan.py).

Ranking (per layer, hottest first): the number of recorded decode steps whose
distinct routed set contains the expert (the distinct-fetch basis: what a UVA
read moves), ties broken by prefill token counts, then by expert id. Every
recorded target step counts (M <= the probe's mmax: c1 verify rows, c8 batches
when the capture used mmax >= 32). Prefill chunks are not in the ring; their
counts only break ties (prefill routing predicts decode badly).

Default layers: the ``--layers`` most concentrated layers inside ``--stage``
(in-sample hit at ``--hot-frac``, per EP owner from the capture's local masks).

Usage (capture -> profile):

    python tools/merlin/cold_expert_profile.py --probe-dir <run>/probe \\
        --config <checkpoint>/config.json --stage 0-24 --layers 4 --hot-frac 0.5 \\
        --name <checkpoint name> --out tools/merlin/profiles/expert_hot/<name>.json

``--config`` ties the profile to the checkpoint (sha256 of config.json; the
launcher and the engine refuse a mismatch). Without it the profile is a
placeholder (allowed, logged loudly at boot).
"""

from __future__ import annotations

import argparse
import array
import glob
import hashlib
import json
import os
import re
import struct
import sys
import time
import zipfile
from collections import defaultdict

VERSION = "cold_expert_profile/1"
FORMAT = "merlin-cold-experts/1"
_DESCR = re.compile(r"'descr':\s*'([^']+)'")
_SHAPE = re.compile(r"'shape':\s*\(([^)]*)\)")


def _npy(z, name):
    """A .npy member of an .npz as (array.array | str, shape); the header is parsed
    with two regexes (never evaluated)."""
    b = z.read(name)
    hl = struct.unpack("<H", b[8:10])[0]
    hdr = b[10:10 + hl].decode("latin-1")
    descr = _DESCR.search(hdr).group(1)
    shape = tuple(int(x) for x in _SHAPE.search(hdr).group(1).replace(" ", "").split(",") if x)
    data = b[10 + hl:]
    if descr.startswith("<U"):
        return data.decode("utf-32-le").rstrip("\x00"), shape
    code = {"<i2": "h", "<i4": "i", "|b1": "b", "<i8": "q"}.get(descr)
    if code is None:
        raise ValueError(f"{name}: dtype {descr} not handled")
    return array.array(code, data), shape


def load(probe_dir: str):
    """-> (experts, topk, segments, owners). segments: {name: {"steps": {layer: [sorted id tuples]},
    "prefill": {layer: [counts]}}}; owners: {layer: [set of expert ids per owning rank]}."""
    files = sorted(glob.glob(os.path.join(probe_dir, "seg-*-rank*-pid*.npz")))
    if not files:
        raise SystemExit(f"no probe segments in {probe_dir}")
    segs: dict = {}
    owners: dict = defaultdict(dict)
    E = K = None
    for f in files:
        m = re.match(r"seg-(\d+)-(.+)-rank(\d+)-pid(\d+)\.npz$", os.path.basename(f))
        if not m:
            continue
        rank = int(m.group(3))
        z = zipfile.ZipFile(f)
        meta = json.loads(_npy(z, "meta.npy")[0])
        name = meta.get("segment") or m.group(2)
        E, K = int(meta["experts"]), int(meta["topk"])
        slots = [s for s in meta.get("slots", []) if s.get("kind") == "target"]
        if not slots:
            continue
        mask = None
        if "local_mask.npy" in z.namelist():
            mk, _ = _npy(z, "local_mask.npy")
            mask = frozenset(e for e in range(E) if mk[e])
        ids, (S, Ls, W) = _npy(z, "target_ids.npy")
        mm, _ = _npy(z, "target_m.npy")
        acc, ashp = _npy(z, "acc_counts.npy")
        seg = segs.setdefault(name, {"steps": {}, "prefill": {}})
        for sl in slots:
            layer, pos, slot = int(sl["layer"]), int(sl["pos"]), int(sl["slot"])
            if mask is not None:
                owners[layer][rank] = mask
            if layer in seg["steps"]:
                continue  # another rank of the stage recorded the same router output
            steps = []
            for s in range(S):
                row = mm[s * Ls:(s + 1) * Ls]
                mx = max(row) if Ls else 0
                if mx <= 0 or min(row) != mx or mx * K > W:
                    continue  # empty, desynced, or wider than the ring (prefill)
                base = s * Ls * W + pos * W
                steps.append(tuple(sorted(set(x for x in ids[base: base + mx * K] if x >= 0))))
            seg["steps"][layer] = steps
            if slot < ashp[0]:
                seg["prefill"][layer] = list(acc[slot * E:(slot + 1) * E])
    groups = {}
    for layer, by_rank in owners.items():
        sets = list(dict.fromkeys(by_rank.values()))
        groups[layer] = [set(s) for s in sets] if sets else None
    return E, K, segs, groups


def freq(segs, E):
    dec = defaultdict(lambda: [0] * E)
    pre = defaultdict(lambda: [0] * E)
    nsteps = 0
    for seg in segs.values():
        for layer, steps in seg["steps"].items():
            f = dec[layer]
            for st in steps:
                for e in st:
                    f[e] += 1
        nsteps += max((len(v) for v in seg["steps"].values()), default=0)
        for layer, c in seg["prefill"].items():
            p = pre[layer]
            for e in range(E):
                p[e] += c[e]
    return dec, pre, nsteps


def ranking(dec, pre, layer, E):
    d, p = dec.get(layer, [0] * E), pre.get(layer, [0] * E)
    return sorted(range(E), key=lambda e: (-d[e], -p[e], e))


def hot_set(rank, groups, E, frac):
    owned = groups or [set(range(E))]
    hot = set()
    for g in owned:
        mine = [e for e in rank if e in g]
        hot.update(mine[: int(round(frac * len(mine)))])
    return hot


def hit(f, hot):
    u = sum(f)
    return 1 - sum(f[e] for e in range(len(f)) if e not in hot) / u if u else None


def parse_stage(raw: str):
    a, _, b = raw.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="checkpoint config.json (ties the profile to it)")
    ap.add_argument("--name", default="")
    ap.add_argument("--stage", default="0-24", help="layer range for the default layers (fn-quad PP0 = 0-24)")
    ap.add_argument("--layers", type=int, default=4, help="number of default layers")
    ap.add_argument("--hot-frac", type=float, default=0.5)
    ap.add_argument("--num-layers", type=int, default=0, help="decoder layers (default: config.json or the capture)")
    a = ap.parse_args(argv)
    if not 0.0 <= a.hot_frac < 1.0:
        raise SystemExit("--hot-frac must be in [0, 1)")
    E, K, segs, groups = load(a.probe_dir)
    dec, pre, nsteps = freq(segs, E)
    layers = sorted(set(dec) | set(pre))
    sha, L = None, a.num_layers
    if a.config:
        with open(a.config, "rb") as f:
            raw = f.read()
        sha = hashlib.sha256(raw).hexdigest()
        cfg = json.loads(raw)
        tc = cfg.get("text_config", cfg)
        L = L or int(tc.get("num_hidden_layers", 0))
        ne = tc.get("num_experts") or tc.get("n_routed_experts")
        if ne and int(ne) != E:
            raise SystemExit(f"config.json has {ne} experts, the capture {E}")
    L = L or (max(layers) + 1)
    target = [x for x in layers if x < L]  # the MTP layer (index L) is not a target layer
    rank = {x: ranking(dec, pre, x, E) for x in target}
    per_layer = {x: hit(dec[x], hot_set(rank[x], groups.get(x), E, a.hot_frac)) for x in target if x in dec}
    stage = [x for x in parse_stage(a.stage) if per_layer.get(x) is not None]
    default = sorted(sorted(stage, key=lambda x: -per_layer[x])[: a.layers])
    profile = {
        "format": FORMAT,
        "num_layers": L,
        "num_experts": E,
        "top_k": K,
        "checkpoint": {"name": a.name, "config_sha256": sha},
        "source": {
            "tool": VERSION, "probe_dir": a.probe_dir.replace("\\", "/"), "segments": sorted(segs),
            "decode_steps": nsteps, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "basis": "decode distinct-step frequency, prefill token counts as tie-break",
            "in_sample_hit_at_default": {str(x): round(per_layer[x], 4) for x in default},
        },
        "default": {"layers": default, "hot_frac": a.hot_frac},
        "ranking": {str(x): rank[x] for x in target},
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(profile, f, separators=(",", ":"))
        f.write("\n")
    print(f"PROFILE {a.out}: {len(target)} layers, {E} experts, {nsteps} decode steps, "
          f"{'checkpoint ' + sha[:12] if sha else 'PLACEHOLDER (no --config)'}")
    print("per-layer in-sample hit at hot_frac %.2f: %s" % (
        a.hot_frac, " ".join(f"L{x}:{per_layer[x]:.3f}" for x in sorted(per_layer))))
    print(f"default layers {default} (stage {a.stage}, top {a.layers} by hit)")
    if nsteps == 0:
        print("WARNING: no decode steps recorded; the ranking is prefill-only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

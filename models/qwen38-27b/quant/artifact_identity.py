#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Artefact identity check: prove the candidate is the tested
checkpoint and differs from the source ONLY in the nine quantised matrices.

  * every file other than config.json, model.safetensors.index.json, model.safetensors,
    model-mtp.safetensors, QUANT-REPORT.txt must be BYTE-IDENTICAL (sha256) to the source
    (tokenizer, chat template, kv_scales.safetensors, generation_config, ...);
  * in the two rewritten shards, every tensor NOT belonging to the nine names must be identical
    (same dtype/shape and same bytes) to the source; the nine names must be present ONLY as
    weight_packed / weight_scale / weight_zero_point / weight_shape and absent as .weight;
  * config.json must differ from the source only in quantization_config.ignore (minus the nine
    names) and quantization_config.config_groups.group_head_mtp (added);
  * prints sha256 of config.json, the index and both shards for the record.

Limits: this is a structural identity check, not a full config-equivalence or numerical check.
It compares config.json only in the fields named above (plus top-level keys), and it does not
look inside the packed int4 tensors; quantisation quality is judged by serving, not here.
QUANT-REPORT.txt is written by quantize_head_mtp.py into the candidate directory; it is a local
build log and is not part of the published checkpoint.

    python3 artifact_identity.py <src> <cand>
"""
import hashlib, json, struct, sys
from pathlib import Path

NAMES = ["lm_head", "mtp.fc",
         "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
         "mtp.layers.0.self_attn.o_proj",
         "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj", "mtp.layers.0.mlp.down_proj"]
REWRITTEN = {"model.safetensors", "model-mtp.safetensors"}
META = {"config.json", "model.safetensors.index.json", "QUANT-REPORT.txt"}


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hdr(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor_bytes_sha(p, meta, base):
    a, b = meta["data_offsets"]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        f.seek(base + a)
        left = b - a
        while left:
            chunk = f.read(min(left, 1 << 24)); h.update(chunk); left -= len(chunk)
    return h.hexdigest()


def main():
    src, cand = Path(sys.argv[1]), Path(sys.argv[2])
    fails = []
    # 1. untouched files byte-identical
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or f.name in META or f.name in REWRITTEN:
            continue
        c = cand / f.name
        if not c.is_file():
            fails.append("missing in candidate: %s" % f.name); continue
        if f.stat().st_size != c.stat().st_size or sha(f) != sha(c):
            fails.append("differs: %s" % f.name)
    extra = {f.name for f in cand.iterdir()} - {f.name for f in src.iterdir()} - META
    if extra:
        fails.append("unexpected files in candidate: %s" % sorted(extra))
    # 2. rewritten shards: non-target tensors identical, targets packed
    quant_names = set()
    for shard in sorted(REWRITTEN):
        hs, bs = hdr(src / shard); hc, bc = hdr(cand / shard)
        for k, m in hs.items():
            if k == "__metadata__":
                continue
            base = k[:-len(".weight")] if k.endswith(".weight") else None
            if base in NAMES:
                if k in hc:
                    fails.append("%s: %s still present" % (shard, k))
                for suf in ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape"):
                    if "%s.%s" % (base, suf) not in hc:
                        fails.append("%s: %s.%s missing" % (shard, base, suf))
                quant_names.add(base)
                continue
            if k not in hc:
                fails.append("%s: %s missing in candidate" % (shard, k)); continue
            if (m["dtype"], m["shape"]) != (hc[k]["dtype"], hc[k]["shape"]):
                fails.append("%s: %s dtype/shape changed" % (shard, k)); continue
            if tensor_bytes_sha(src / shard, m, bs) != tensor_bytes_sha(cand / shard, hc[k], bc):
                fails.append("%s: %s bytes changed" % (shard, k))
        for k in hc:
            if k == "__metadata__" or k in hs:
                continue
            base = k.rsplit(".", 1)[0]
            if base not in NAMES:
                fails.append("%s: unexpected new tensor %s" % (shard, k))
    if quant_names != set(NAMES):
        fails.append("quantised set mismatch: %s" % sorted(set(NAMES) ^ quant_names))
    # 3. config diff
    cs = json.load(open(src / "config.json", encoding="utf-8"))
    cc = json.load(open(cand / "config.json", encoding="utf-8"))
    qs, qc = cs["quantization_config"], cc["quantization_config"]
    if sorted(qc["ignore"]) != sorted(x for x in qs["ignore"] if x not in NAMES):
        fails.append("config: ignore list differs beyond the nine names")
    g = dict(qc["config_groups"]); grp = g.pop("group_head_mtp", None)
    if g != qs["config_groups"]:
        fails.append("config: config_groups differ beyond group_head_mtp")
    if not grp or set(NAMES) - set(grp["targets"]):
        fails.append("config: group_head_mtp missing or does not target the nine names")
    for k in set(cs) | set(cc):
        if k == "quantization_config":
            continue
        if cs.get(k) != cc.get(k):
            fails.append("config: top-level key differs: %s" % k)
    for f in ("config.json", "model.safetensors.index.json", "model.safetensors", "model-mtp.safetensors"):
        print("sha256 %s %s" % (sha(cand / f), f))
    for x in fails:
        print("FAIL", x)
    print("IDENTITY %s: %d checks failed" % ("OK" if not fails else "FAILED", len(fails)))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

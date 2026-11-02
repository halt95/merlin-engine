#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Merge absmax.*.json (max across ranks/processes) -> per-layer static E4M3 scales -> new checkpoint dir
(hardlinked shards + kv_scales.safetensors + index + config kv_cache_scheme).
usage: kvcal_write.py <absmax_dir> <src_ckpt> <dst_ckpt> [--margin 1.0]"""
import glob, json, os, sys, shutil
import torch
from safetensors.torch import save_file

absdir, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
margin = float(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 1.10   # never map the observed max exactly onto the E4M3 ceiling
assert 1.0 <= margin <= 2.0, f"margin {margin} out of range"
E4M3_MAX = 448.0
merged = {}
files = sorted(glob.glob(os.path.join(absdir, "absmax.*.json")))
assert files, "no absmax files"
for f in files:
    for name, d in json.load(open(f, encoding="utf-8")).items():
        m = merged.setdefault(name, {"k": 0.0, "v": 0.0, "calls": 0, "files": 0})
        m["k"] = max(m["k"], d["k_absmax"]); m["v"] = max(m["v"], d["v_absmax"]); m["calls"] += d["calls"]; m["files"] += 1
print(f"{len(files)} absmax files, {len(merged)} attention layers")
assert len(merged) == 17, f"expected 16 attention + 1 MTP layers, got {len(merged)}"
assert sum(n.startswith("mtp.") for n in merged) == 1, "MTP draft layer missing from the calibration"
import math
assert all(math.isfinite(m["k"]) and math.isfinite(m["v"]) and m["k"] > 0 and m["v"] > 0 for m in merged.values()), "non-finite or zero absmax"
tensors = {}
report = []
for name in sorted(merged, key=lambda n: (not n.startswith("model."), n)):
    m = merged[name]
    assert name.endswith(".attn"), name
    base = name[: -len(".attn")]                      # vLLM-internal: language_model.model.layers.3.self_attn / mtp.layers.0.self_attn
    if base.startswith("language_model.model.layers."):
        base = "model.language_model.layers." + base[len("language_model.model.layers."):]   # checkpoint (HF) name
    elif not base.startswith("mtp."):
        raise SystemExit(f"unexpected layer name {name}")
    ks = max(m["k"], 1e-6) * margin / E4M3_MAX; vs = max(m["v"], 1e-6) * margin / E4M3_MAX
    tensors[base + ".k_scale"] = torch.tensor(ks, dtype=torch.float32)
    tensors[base + ".v_scale"] = torch.tensor(vs, dtype=torch.float32)
    report.append((base, m["k"], m["v"], ks, vs, m["calls"], m["files"]))
for r in report:
    print(f"{r[0]:<55} k_absmax {r[1]:8.3f} v_absmax {r[2]:8.3f} k_scale {r[3]:.5f} v_scale {r[4]:.5f} calls {r[5]} files {r[6]}")
if "--dry" in sys.argv:
    sys.exit(0)
if os.path.exists(dst) and os.listdir(dst):
    raise SystemExit(f"refusing to write into non-empty {dst}")
os.makedirs(dst, exist_ok=True)
assert not os.path.exists(os.path.join(src, "kv_scales.safetensors")), "source already has kv_scales.safetensors"
for f in os.listdir(src):
    s = os.path.join(src, f); d = os.path.join(dst, f)
    if os.path.exists(d):
        continue
    if f.endswith(".safetensors"):
        os.link(s, d)
    elif os.path.isfile(s):
        shutil.copy2(s, d)
save_file(tensors, os.path.join(dst, "kv_scales.safetensors"), metadata={"format": "pt"})
idx = json.load(open(os.path.join(src, "model.safetensors.index.json"), encoding="utf-8"))
for k in tensors:
    idx["weight_map"][k] = "kv_scales.safetensors"
idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + 4 * len(tensors)
json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w", encoding="utf-8"), indent=2)
cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))
cfg["quantization_config"]["kv_cache_scheme"] = {"dynamic": False, "num_bits": 8, "observer": "minmax", "observer_kwargs": {}, "strategy": "tensor", "symmetric": True, "type": "float"}
json.dump(cfg, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
json.dump({r[0]: {"k_absmax": r[1], "v_absmax": r[2], "k_scale": r[3], "v_scale": r[4], "calls": r[5]} for r in report}, open(os.path.join(dst, "kv_scales_report.json"), "w"), indent=1)
print("wrote", dst, len(tensors), "scale tensors")

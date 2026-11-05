#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Post-hoc GPTQ int4 (group 128, ASYMMETRIC, minmax per group) of the shared `lm_head`
and the MTP drafter's linears on the already-quantised philbert checkpoint, using the Hessians
captured by quant/mtpcal.py from the served model.

Everything else in the checkpoint is copied byte-for-byte (untouched shards are hard-linked).
Only two shards are rewritten: `model.safetensors` (lm_head.weight -> lm_head.weight_packed /
weight_scale / weight_zero_point / weight_shape) and `model-mtp.safetensors` (the eight mtp.*
matrices, norms kept BF16). config.json: the nine names leave `ignore`, and a new config group
`group_head_mtp` targets exactly those nine names with the SAME W4A16 asym g128 scheme as
group_0 - so vLLM's CompressedTensorsWNA16 (Marlin, min capability 75, uint4 + zero points)
serves them. Nothing about the loader needs patching: the drafter's linears are named `mtp.*`
at runtime and `lm_head` matches the explicit target (ParallelLMHead is not a `Linear`).

GPTQ here is the standard column-wise error-feedback quantiser (Frantar et al.) with damp 1 %
and Cholesky inverse, rows processed in chunks because rows are independent given H; asym
per-group minmax scale/zero-point (matches philbert's asym g128 recipe; the MSE observer of the
original AWQ pass is not reproduced).

    python3 quantize_head_mtp.py <src_ckpt> <dest_ckpt> --hessians ./mtpcal-out/hessians.pt [--dry-run]
"""
import argparse, json, math, os, shutil, struct, sys, time
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors import safe_open
from safetensors.torch import save_file

GS = 128
BITS = 4
TARGETS = ["lm_head", "mtp.fc",
           "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
           "mtp.layers.0.self_attn.o_proj",
           "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj", "mtp.layers.0.mlp.down_proj"]
# checkpoint tensor -> the Hessian key mtpcal recorded (fused modules share their input)
HKEY = {"lm_head": ("lm_head@target", "lm_head@draft"), "mtp.fc": ("mtp.fc",),
        "mtp.layers.0.self_attn.q_proj": ("mtp.layers.0.self_attn.qkv_proj",),
        "mtp.layers.0.self_attn.k_proj": ("mtp.layers.0.self_attn.qkv_proj",),
        "mtp.layers.0.self_attn.v_proj": ("mtp.layers.0.self_attn.qkv_proj",),
        "mtp.layers.0.self_attn.o_proj": ("mtp.layers.0.self_attn.o_proj",),
        "mtp.layers.0.mlp.gate_proj": ("mtp.layers.0.mlp.gate_up_proj",),
        "mtp.layers.0.mlp.up_proj": ("mtp.layers.0.mlp.gate_up_proj",),
        "mtp.layers.0.mlp.down_proj": ("mtp.layers.0.mlp.down_proj",)}
# Runtime prefixes differ from checkpoint names for the TARGET head: Qwen3_5ForConditionalGeneration
# builds its language model under "language_model.", so its head is "language_model.lm_head" at
# quant-dispatch time, so the checkpoint name alone is not targeted. The drafter's head is "lm_head" (prefix "").
GROUP = {"format": "pack-quantized", "input_activations": None, "output_activations": None,
         "targets": list(TARGETS) + ["language_model.lm_head", "model.language_model.lm_head"],
         "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": GS,
                     "num_bits": BITS, "observer": "minmax", "observer_kwargs": {}, "scale_dtype": None,
                     "strategy": "group", "symmetric": False, "type": "int", "zp_dtype": "torch.int8"}}
SCHEME = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
    num_bits=BITS, type="int", strategy="group", group_size=GS, symmetric=False, dynamic=False))


def hdr(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


# ----------------------------------------------------------------------------- GPTQ
def gptq_asym_group(W: torch.Tensor, H: torch.Tensor, damp=0.01, block=128):
    """W [out, in] fp32 on GPU, H [in, in] fp32. Returns q (uint8 codes 0..15) [out,in],
    scale [out, in/GS] fp32, zp (0..15) [out, in/GS] uint8. Standard GPTQ with asym minmax per
    group, groups quantised when their first column is reached (no act-order)."""
    out, inn = W.shape
    assert inn % GS == 0 and inn % block == 0 and block % GS == 0
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W = W.clone()
    W[:, dead] = 0
    H.diagonal().add_(damp * torch.mean(torch.diag(H)))
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper Cholesky of H^-1, as in GPTQ
    Q = torch.zeros_like(W, dtype=torch.uint8)
    scale = torch.zeros(out, inn // GS, dtype=torch.float32, device=W.device)
    zp = torch.zeros(out, inn // GS, dtype=torch.uint8, device=W.device)
    maxq = 2 ** BITS - 1
    for i1 in range(0, inn, block):
        i2 = i1 + block
        W1 = W[:, i1:i2].clone(); Err1 = torch.zeros_like(W1); Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(block):
            col = i1 + i
            if col % GS == 0:
                g = col // GS
                wg = W[:, col:col + GS]  # current (error-updated) group window
                wmin = wg.amin(dim=1).clamp(max=0); wmax = wg.amax(dim=1).clamp(min=0)
                s = ((wmax - wmin) / maxq).clamp_min(1e-8)
                z = torch.round(-wmin / s).clamp(0, maxq)
                scale[:, g] = s; zp[:, g] = z.to(torch.uint8)
            g = col // GS
            s = scale[:, g]; z = zp[:, g].float()
            w = W1[:, i]
            q = torch.clamp(torch.round(w / s) + z, 0, maxq)
            Q[:, col] = q.to(torch.uint8)
            wq = (q - z) * s
            d = Hinv1[i, i]
            err = (w - wq) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scale, zp


def quantize_matrix(name, W_bf16, H, device, rows_per_chunk):
    out, inn = W_bf16.shape
    assert H.shape == (inn, inn), "%s: Hessian %s does not match weight %s" % (name, tuple(H.shape), tuple(W_bf16.shape))
    Hd = H.to(device)
    Qs, Ss, Zs = [], [], []
    for r0 in range(0, out, rows_per_chunk):
        W = W_bf16[r0:r0 + rows_per_chunk].to(device=device, dtype=torch.float32)
        q, s, z = gptq_asym_group(W, Hd)
        Qs.append(q.cpu()); Ss.append(s.cpu()); Zs.append(z.cpu())
        del W, q, s, z
    Q = torch.cat(Qs); S = torch.cat(Ss); Z = torch.cat(Zs)
    # dequant and error report (plain, and H-weighted = what GPTQ minimises)
    Wf = W_bf16.float()
    deq = ((Q.float() - Z.float().repeat_interleave(GS, dim=1)) * S.repeat_interleave(GS, dim=1))
    rel = float((deq - Wf).norm() / Wf.norm())
    E = (deq - Wf).to(device)
    hrel = float(torch.sqrt((E @ Hd * E).sum() / ((Wf.to(device) @ Hd * Wf.to(device)).sum())))
    del E
    return Q, S, Z, rel, hrel


def pack_ct(prefix, Q, S, Z, shape):
    """compressed-tensors pack-quantized serialisation for an asym int4 group scheme."""
    # compressed-tensors convention (int type, 4 bits, symmetric OR asymmetric): codes live in
    # the SIGNED range [-8, 7] and so does the zero point; dequant = (q - zp) * scale.
    # pack_to_int32 adds 2^(bits-1) before packing, so unsigned 0..15 codes overflow the nibble.
    # Handing it 0..15 makes the packed nibbles overflow.
    # Our minmax codes/zp are 0..15 -> shift both by -8 (dequant unchanged).
    w_int = (Q.to(torch.int16) - 8).to(torch.int8)
    zp_int = (Z.to(torch.int16) - 8).to(torch.int8)
    # NOT PackedQuantizationCompressor.compress(): that takes the FLOAT weight and re-quantises
    # it with its own observer (feeding it int8 codes returns different codes).
    # Our codes are GPTQ's; pack them directly with the same primitive compress() ends in.
    out = {
        "%s.weight_packed" % prefix: pack_to_int32(w_int, BITS),
        "%s.weight_scale" % prefix: S.to(torch.bfloat16),
        "%s.weight_zero_point" % prefix: pack_to_int32(zp_int, BITS, packed_dim=0),
        "%s.weight_shape" % prefix: torch.tensor(list(shape), dtype=torch.int64),  # philbert: int64
    }
    out = {k: v.cpu().contiguous() for k, v in out.items()}
    # round-trip through CT's own unpacker: the packed codes must come back as w_int exactly
    back = unpack_from_int32(out["%s.weight_packed" % prefix], BITS, torch.Size(shape))
    assert torch.equal(back.to(torch.int8), w_int), "%s: packed codes do not round-trip" % prefix
    zback = unpack_from_int32(out["%s.weight_zero_point" % prefix], BITS, torch.Size(Z.shape), packed_dim=0)
    assert torch.equal(zback.to(torch.int8), zp_int), "%s: packed zero points do not round-trip" % prefix
    return out


def load_hessian(hs, name):
    keys = HKEY[name]
    H = None; n = 0
    for k in keys:
        if k not in hs["H"]:
            raise SystemExit("hessians.pt lacks %s (needed for %s)" % (k, name))
        H = hs["H"][k].clone() if H is None else H + hs["H"][k]
        n += hs["n"][k]
    return H, n


def patch_config(cfg):
    qc = cfg["quantization_config"]
    missing = [t for t in TARGETS if t not in qc["ignore"]]
    assert not missing, "not in ignore (already quantised?): %s" % missing
    qc["ignore"] = [x for x in qc["ignore"] if x not in TARGETS]
    assert "group_head_mtp" not in qc["config_groups"]
    qc["config_groups"]["group_head_mtp"] = GROUP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path); ap.add_argument("dest", type=Path)
    ap.add_argument("--hessians", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows-per-chunk", type=int, default=16384)
    ap.add_argument("--dry-run", action="store_true", help="quantise + report errors, write nothing")
    ap.add_argument("--only", default=None, help="comma list of target names (dry-run subsets)")
    a = ap.parse_args()
    src, dest = a.src.resolve(), a.dest.resolve(); assert src != dest
    if a.only and not a.dry_run:
        sys.exit("--only builds a partial checkpoint; use it with --dry-run only")
    if dest.exists() and not a.dry_run:
        sys.exit("destination exists: %s" % dest)
    hs = torch.load(a.hessians, map_location="cpu", weights_only=True)
    print("hessians:", {k: hs["n"][k] for k in sorted(hs["n"])}, flush=True)
    for k in ("lm_head@target", "lm_head@draft", "mtp.fc", "mtp.layers.0.mlp.down_proj"):
        if hs["n"].get(k, 0) < 100_000:
            print("WARNING: only %d calibration rows for %s (want >= 100k)" % (hs["n"].get(k, 0), k))

    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8")); wm = index["weight_map"]
    targets = [t for t in TARGETS if (not a.only or t in a.only.split(","))]
    packed = {}   # tensor name -> tensor, per shard
    report = []
    t0 = time.time()
    for name in targets:
        key = name + ".weight"; shard = wm[key]
        with safe_open(src / shard, framework="pt", device="cpu") as h:
            W = h.get_tensor(key)
        H, n = load_hessian(hs, name)
        Q, S, Z, rel, hrel = quantize_matrix(name, W, H, a.device, a.rows_per_chunk)
        p = pack_ct(name, Q, S, Z, W.shape)
        packed.setdefault(shard, {}).update(p)
        nbytes = sum(t.numel() * t.element_size() for t in p.values())
        line = "%-36s %-14s rows=%-7d rel-fro %.4f  H-weighted %.4f  %.3f GB -> %.3f GB  (%.0fs)" % (
            name, str(list(W.shape)), n, rel, hrel, W.numel() * 2 / 1e9, nbytes / 1e9, time.time() - t0)
        print(line, flush=True); report.append(line)
        del W, Q, S, Z
    if a.dry_run:
        print("dry run - nothing written"); return

    tmp = dest.parent / (".%s-building" % dest.name); tmp.mkdir(parents=True, exist_ok=False)
    new_wm = dict(wm)
    for item in sorted(src.iterdir()):
        if item.name in ("config.json", "model.safetensors.index.json") or item.name.startswith("."):
            continue
        if item.name in packed:
            with safe_open(item, framework="pt", device="cpu") as h:
                meta = h.metadata(); tensors = {}
                drop = {t + ".weight" for t in targets if wm[t + ".weight"] == item.name}
                for k in h.keys():
                    if k in drop:
                        continue
                    tensors[k] = h.get_tensor(k)
            for k in drop:
                del new_wm[k]
            tensors.update(packed[item.name])
            for k in packed[item.name]:
                new_wm[k] = item.name
            save_file(tensors, tmp / item.name, metadata=meta or {"format": "pt"})
            print("wrote", item.name, flush=True)
        else:
            try:
                os.link(item, tmp / item.name)
            except OSError:
                shutil.copy2(item, tmp / item.name)
    cfg = json.load(open(src / "config.json", encoding="utf-8")); patch_config(cfg)
    json.dump(cfg, open(tmp / "config.json", "w", encoding="utf-8"), indent=2)
    total = 0
    for f in sorted(set(new_wm.values())):
        for k, v in hdr(tmp / f).items():
            if k != "__metadata__":
                total += v["data_offsets"][1] - v["data_offsets"][0]
    index["weight_map"] = new_wm; index.setdefault("metadata", {})["total_size"] = total
    json.dump(index, open(tmp / "model.safetensors.index.json", "w", encoding="utf-8"), indent=2)
    (tmp / "QUANT-REPORT.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    tmp.rename(dest)
    print("done ->", dest, "total_size", total)


if __name__ == "__main__":
    main()

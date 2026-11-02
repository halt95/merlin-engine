# Merlin engine hardware matrix (v3)

Applies to Flash-Next and the other recipes on the Merlin engine (Merlin 3.0.1, vLLM 0.32 base).

This page says, for every GPU platform the v3 engine is meant to run on: the layout to use, the concurrency cap, how much
context fits, whether peer-to-peer (P2P) is needed, and **how much evidence there is**. Every row carries one of four labels:

- **tested on the reference host**: run and checked on the reference host (4 x RTX 3090, 220 W, PCIe Gen4 x16, P2P over the open-kernel-module
  patch, 96 GiB (98,304 MiB) RAM allocation). Which rows carry it: the Label and Status columns below.
- **🟡 testing**: full-model serving runs passed on cloud-hosted hardware of that kind (boot and pool, 262K needles, concurrency, image,
  GSM8K), but kernel parity qualification has not run. On other cards that resolve to the row it is reported untested
  (`match.testing_gpu_names`). `merlin serve` picks a testing row exactly like an untested one.
- **parked, not in this release**: no row at present.
- **🟠 partly tested** (only in the README and how-to-serve summaries, never on a row): a hardware class where some models or layouts were measured and others weren't; each row below carries its own label.
- **untested / community-validated**: shipped so that it can be tried; the full matrix qualification has **not been completed by the maintainers** (targeted launcher checks may have passed). Not every platform can be
  tested here. A row moves to "tested" only when a run on v3 reports its pool, decode at
  4K / 131K / 262K, the needles and the smoke test. Until then expect rough edges and report what you see.

A leading `~` in a number means an **estimate** (see "Where the numbers come from"). Layouts, admission caps, pins and software version numbers describe configuration, not measurements. The provenance and limits of the capacity numbers are listed below.

## At a glance

| Platform | Model | Layout (profile) | max-num-seqs | Context fit (pool) | P2P | Label |
|---|---|---|---|---|---|---|
| 4 x RTX 3090 24 GB (the reference host) | Flash-Next | TP2xPP2+EP, partition 25,23 (`fn-quad-3090`) | 8 | 3 x 262K or 6 x 131K (924,993) | needed for the tested numbers | **tested on the reference host** |
| 4 x RTX 3090, no P2P | Flash-Next | the same launcher (`fn-quad-3090-nop2p`, the default); PP4 x TP1 as the named prefill alternative (`--profile fn-quad-3090-nop2p-pp4`, 250-block pin: c1 prefill +47-53 %, real-text c1 decode -14 % against the default) | 8 (32) | ~3 x 262K / ~6 x 131K (PP4: 3 x / 6 x, 906,581 measured) | off | default: untested / community-validated; PP4: **tested on the reference host with P2P off** (2026-10-19) |
| **8 x RTX 3090 24 GB (octo)** | Flash-Next | **TP2xPP4+EP, partition 12,12,12,12** (`fn-octo-3090-tp2pp4`); TP4xPP2+EP as the latency alternative (`fn-octo-3090-tp4pp2`) | **32** (20) | untested registry estimates: ~17 x 262K / ~31 x 131K (~4.66M; with the 11.55 GiB pin ~32, ~4.78M); alt ~11 / ~20 (~2.96M) | on or off | untested / community-validated |
| 2 x 48 GB sm_86 (RTX A6000, A40) | Flash-Next | PP2 x TP1, fp8 KV default, bf16 selectable (`fn-2x48g-pp2-sm86`) | 9 | ~5 x 262K / ~9 x 131K | any | untested / community-validated |
| 4 x 48 GB sm_86 | Flash-Next | TP2xPP2+EP, fp8 KV default, bf16 selectable (`fn-4x48g-tp2pp2-sm86`) | 32 | ~25 / ~46 | any | untested / community-validated |
| 4 x NVIDIA A10 / A10G 22.5 GiB as reported (sm_86; AWS g5.12xlarge, g5.24xlarge) | Flash-Next | the no-P2P quad (`fn-quad-3090-nop2p`) with the **KV pin lowered by the ~1.5 GiB these cards lack against a 3090** (preflight's runtime estimate at 22.49 GiB reported VRAM: 2.31 GiB per card, `KV_BYTES`, printed by preflight; registry profile pin 3.82 GiB / 4.1e9 bytes) | 8 | estimate at 22.49 GiB reported VRAM: ~559K pool, ~1 x 262K / ~3 x 131K (untested, linear from the sizing); registry profile pool ~924,993 (~3 / 6) | off (AWS does not document P2P; probe with `nvidia-smi topo -p2p r`) | untested / community-validated |
| 8 x NVIDIA A10G 22.5 GiB (sm_86; AWS g5.48xlarge) | Flash-Next | octo TP2xPP4+EP (`fn-octo-3090-tp2pp4`) with the pin lowered by preflight to 9.74 GiB (estimate at 22.49 GiB reported VRAM; registry profile pin 11.25 GiB) | 32 | estimate at 22.49 GiB reported VRAM: ~4.03M pool, ~14 x 262K / ~26 x 131K (untested); registry profile pool ~4,656,696 (~17 / 31) | on or off | untested / community-validated |
| 4 x NVIDIA L4 22.5 GiB (sm_89; AWS g6.12xlarge, GCP g2-standard-48) | Flash-Next | the sm_89 quad (`fn-quad-24g-sm89`), fp8 KV default, bf16 selectable; preflight lowers the pin to 2.31 GiB (estimate at 22.49 GiB reported VRAM; registry profile pin 3.82 GiB) | 8 | estimate at 22.49 GiB reported VRAM: ~560K pool, ~1 x 262K / ~3 x 131K (untested); registry profile pool ~924,993 (~3 / 6) | off by default | untested / community-validated |
| 4 x RTX A5000 / RTX 3090 Ti / other sm_86 24 GB (24,564 MiB) | Flash-Next, 27B, Super VL, Ornith 1.5 | the 3090 rows (`fn-quad-3090`, `27b-tp2-3090`, `27b-tp4-3090`, `svl-quad-3090`, `ornith-quad-3090`); matching is by arch + VRAM class + P2P, not by name | as the 3090 rows | as the 3090 rows | needed for the tested numbers | **untested / community-validated** (the "tested on the reference host" label applies to the RTX 3090 only) |
| 4 x RTX 4090 24 GB (sm_89) | Flash-Next | the quad layout, **fp8 KV default**, bf16 selectable (`fn-quad-24g-sm89`, the default); PP4 x TP1 as the named no-P2P prefill alternative (`--profile fn-quad-24g-sm89-pp4`, 2.5e9 pin; ~3x c1 prefill on cloud-hosted RTX 4090s) | 8 (bf16: 3; PP4: 32) | ~3 x 262K / ~6 x 131K (bf16: ~1 / ~3; PP4: ~1 / ~3); the cloud-hosted cards booted the fp8 pin to 924,993 tokens | 4090: off by default | fp8: 🟡 testing (cloud-hosted 4 x RTX 4090, 2026-10-18; at chunk 1024); bf16 and PP4: untested / community-validated |
| 4 x A30 24 GB (sm_80) | Flash-Next | the quad layout, **BF16 KV default**; FP8 opt-in via `KV_DTYPE=fp8` (`fn-quad-24g-sm80`; byte decoder) | 3 | BF16: ~1 x 262K / ~3 x 131K; FP8 engine-fit estimate: ~921,248-token pool, ~3 / ~6 (not measured) | off by default; on if the driver reports P2P | BF16: untested / community-validated; FP8: ⚫ untested |
| 2 / 4 x 48 GB Ada (RTX 6000 Ada, L40S) | Flash-Next | PP2 x TP1 (`fn-2x48g-pp2-sm89`); TP2xPP2+EP (`fn-4x48g-tp2pp2-sm89`); fp8 KV default, bf16 selectable | 9 / 32 (bf16: 4 / 8) | ~5 / ~9 and ~25 / ~46 (bf16: ~2 / ~4 and ~13 / ~24) | off by default | untested / community-validated |
| 4 x 32 GB Ada (modified RTX 4080 Super) | Flash-Next | TP2xPP2+EP, fp8 KV default, bf16 selectable (`fn-quad-32g-sm89`) | 19 (bf16: 8) | ~10 / ~19 (bf16: ~5 / ~10) | off by default | untested / community-validated |
| 1 / 2 / 4 x A100 80 GB (sm_80) | Flash-Next | TP1 (`fn-sm80-80g-x1`); TP2+EP or PP2 (`fn-sm80-80g-x2-tp2`, `-pp2`); TP2xPP2+EP (`fn-sm80-80g-x4-tp2pp2`); **BF16 KV default**, FP8 opt-in via `KV_DTYPE=fp8` | 2 / 20 (PP2: 19) / 32 | BF16: ~1 / ~2; ~11 / ~20 (PP2: ~10 / ~19); ~28 / ~52. FP8 engine-fit estimates (not measured): ~670,339 (~2 / ~4); ~5,632,351 (~21 / ~39; PP2: ~5,542,473, ~21 / ~38); ~14,552,736 (~55 / ~102) | `-x1`: n/a (one card); `-x2-tp2` and `-x4-tp2pp2`: on by default (NVLink / PCIe), off selectable; `-x2-pp2`: off only | BF16: untested / community-validated; FP8: ⚫ untested |
| 2 x CMP 170HX 64 GB (sm_80) | Flash-Next | PP2 x TP1, **BF16 KV default**, FP8 opt-in via `KV_DTYPE=fp8` (`fn-sm80-64g-x2-pp2`) | 12 | BF16: ~6 x 262K / ~12 x 131K (1,790,069-token measured pool on a 4-card host); FP8 engine-fit estimate: ~3,475,280 tokens, ~13 / ~24 (not measured) | off (stock) | BF16: 🟡 testing (cloud-hosted 4 x CMP 170HX 64 GB, 2026-10-19); FP8: ⚫ untested |
| 3 or 4 x CMP 170HX 32 GB, 4 x 40 GB | Flash-Next | PP3 / PP4 x TP1, **BF16 KV default**, FP8 opt-in via `KV_DTYPE=fp8` (`fn-sm80-32g-x3-pp3`, `-x4-pp4`, `fn-sm80-40g-x4-pp4`) | 3 / 8 / 14 | BF16: ~1 / ~4 / ~7 x 262K (40 GB PP4 pool 2,122,638 booted on 64 GB cards); FP8 engine-fit estimates (not measured): ~913,863 / ~2,370,218 / ~4,121,486 tokens, ~3 / ~9 / ~15 x 262K | off (stock) | BF16: untested / community-validated; FP8: ⚫ untested |
| 1 x RTX 3090 24 GB | 27B | TP1 with vision (`27b-tp1-3090`) | 2 | 1 x 262K or 2 x 131K (271,506; measured at the row's release check, 2026-10-21); 4 MP images with the single-card memory settings (558-563 MiB free at 4 MP, 2026-10-21); 8 MP does not fit on one card | n/a | **tested on the reference host** (2026-10-21) |
| 2 x RTX 3090 24 GB | 27B | TP2 (`27b-tp2-3090`) | 16 | 3 x 262K or 6 x 131K (826,689) | needed for the tested numbers | **tested on the reference host** |
| 4 x RTX 3090 24 GB | 27B | TP4 (`27b-tp4-3090`), RecoverSSM on, block 1024, chunk 4096 | 24 | 7 x 262K or 14 x 131K (2,012,105) | needed for the tested numbers | **tested on the reference host** |
| 1 x RTX 4090 24 GB (sm_89) | 27B | TP1 (`27b-tp1-sm89-24g`) | 2 | ~1 x 262K / ~2 x 131K (registry estimate, ~271,506; same per-card KV pin as the 3090 row, extra VRAM is headroom) | n/a | untested / community-validated |
| 2 x RTX 4090 24 GB (sm_89) | 27B | TP2 (`27b-tp2-sm89-24g`) | 16 | ~3 x 262K / ~6 x 131K (826,689; same per-card KV pin as the 3090 row, extra VRAM is headroom; the cloud-hosted cards booted the same pool) | off (consumer Ada has no peer access by default; the launcher takes `P2P=on\|off`) | 🟡 testing (cloud-hosted 4 x RTX 4090, 2026-10-18) |
| 1 x modified RTX 4080 32 GB (sm_89) | 27B | TP1 (`27b-tp1-sm89-32g`) | 2 | ~1 x 262K / ~2 x 131K (registry estimate, ~271,506; same per-card KV pin as the 3090 row, extra VRAM is headroom) | n/a | untested / community-validated |
| 2 x modified RTX 4080 32 GB (sm_89) | 27B | TP2 (`27b-tp2-sm89-32g`) | 16 | ~3 x 262K / ~6 x 131K (registry estimate, 826,689; same per-card KV pin as the 3090 row, extra VRAM is headroom) | off (consumer Ada has no peer access by default; the launcher takes `P2P=on\|off`) | untested / community-validated |
| 1 x modified RTX 4090 48 GB (sm_89) | 27B | TP1 (`27b-tp1-sm89-48g`) | 2 | ~1 x 262K / ~2 x 131K (registry estimate, ~271,506; same per-card KV pin as the 3090 row, extra VRAM is headroom) | n/a | untested / community-validated |
| 4 x RTX 3090 24 GB | Nemotron Super VL | TP2xPP2+EP, RecoverSSM, MTP K=3, layer partition 46,42 (`svl-quad-3090`) | 9 | 1 x 1M or 5 x 262K (1,856,367 at the PROVISIONAL 2.50 GiB pin with the shipped 1M default; 1M residency proven by an identical 1,048,052-token re-send served from cache); 8 MP images accepted, downscaled by the model processor to ~3.4 MP | needed for the tested numbers | **tested on the reference host** (release bench card) |
| 4 x RTX 3090 24 GB | Ornith 1.5 | TP2xPP2+EP, RecoverSSM, MTP K=3 + uniform verify, layer partition 25,23 (`ornith-quad-3090`) | 8 | 3 x 262K (894,618 at the 3,556,769,792-byte pin) | needed for the tested numbers | **tested on the reference host** |
| 4 x RTX 3090, no P2P | Ornith 1.5 | the same recipe without P2P (`ornith-quad-3090-nop2p`) | 8 | 3 x 262K (977,838 at the 3,892,314,112-byte pin) | off | tested on a 4 x RTX 3090 host without P2P (PCIe x8, 2026-10-21) |

**Does not fit (the preflight says so instead of failing at boot):** Flash-Next on 12 GB cards (RTX 3080 Ti, RTX 3080 12 GB: the octo binding stage needs ~10 GiB of weights + ~2.75 GiB overhead = ~12.75 GiB before any KV) and on 20 GB cards (RTX A4500: the quad needs ~17.4 + ~2.75 = ~20.2 GiB before any KV; an 8 x 20 GB octo would leave a ~7 GiB pin by that arithmetic but the registry has no 20 GB class and none is added without a pool-only boot); 8 x 24 GB sm_89 (8 x L4, g6.48xlarge, g2-standard-96: the octo rows are sm_86 only; use `--gpus` to take four cards for the sm_89 quad); and, as before: Flash-Next on 2 x 24 GB (it would need a host-RAM expert tier, which is not
implemented); on any single card below 80 GB; on 2 x 40 GB (no 262K session in the registry capacity model); on 3 x 24 GB; on hosts whose cards form no homogeneous set with a row (same-class mixes such as RTX 3090 + RTX 3090 Ti are one class; other mixes get the largest homogeneous subset); on sm_90 or newer (out of
the v3 target).

Swift1.5 and Swift1.5-Huihui use the same quad layout with separate checkpoint profiles.
Their canonical launchers are `models/swift1.5/serve.sh` and `models/swift1.5-huihui/serve.sh` (generated from the registry; see the commands below).
Both retain checkpoint-specific KV scales and full-vocabulary MTP K=3, and serve `medium` reasoning by default like every checkpoint; these are registry settings.
FGPC is off for all generated Flash-Next-family launchers; the default image cap is 8,388,608 pixels.
The two checkpoint pools are the quad's 924,993 tokens; the launcher qualification measured 924,993 tokens on both generated launchers on 2026-10-16. Their 3 x 262K / 6 x 131K residency in the table below is the inherited quad target, not a separate measurement.

Regenerate the canonical recipes from the repository root:

```bash
python tools/merlin/preflight.py --emit launcher --profile fn-swift1.5-quad-3090 > models/swift1.5/serve.sh
python tools/merlin/preflight.py --emit launcher --profile fn-swift1.5-huihui-quad-3090 > models/swift1.5-huihui/serve.sh
bash models/swift1.5/serve.sh /path/to/checkpoint
```

Use `--profile` to choose a checkpoint explicitly; automatic hardware selection still chooses Flash-Next.

## Consumer Ada (sm_89) shapes by checkpoint

All rows below are **untested / community-validated** except the 4 x RTX 4090 24 GB rows whose full-model runs passed on cloud-hosted cards (`fn-quad-24g-sm89`, `fn-swift1.5-quad-24g-sm89`, `27b-tp4-sm89-24g`, `27b-tp2-sm89-24g`, `svl-quad-sm89-24g`: 🟡 testing); nothing is claimed for throughput. Consumer Ada has no P2P by default, so every row defaults to P2P off (NCCL through host memory, custom all-reduce off) and `on` only when `nvidia-smi topo -p2p r` reports it. The modified 32 GB RTX 4080 and 48 GB RTX 4090 report ~32,760 / ~49,140 MiB, inside the 32 / 48 GiB classes. Build with `TORCH_CUDA_ARCH_LIST` including 8.9 and expect a live first-boot kernel compile: there is no persisted Triton cache on sm_89, and the GDN prefill kernels run fixed launch configurations on sm_89 as on every sm_8x card (see "Other platforms").

| Shape | Flash-Next | Swift 1.5 / Huihui (`--profile`) | 27B (TP4 / TP2 / TP1) | Super VL |
|---|---|---|---|---|
| 4 x RTX 4090 24 GB | `fn-quad-24g-sm89` | `fn-swift1.5-quad-24g-sm89`, `fn-swift1.5-huihui-quad-24g-sm89` | `27b-tp4-sm89-24g`, `27b-tp2-sm89-24g`, `27b-tp1-sm89-24g` (`--gpus` picks 4, 2 or 1 cards) | `svl-quad-sm89-24g` |
| 4 x RTX 4080 32 GB (modified) | `fn-quad-32g-sm89` | `fn-swift1.5-quad-32g-sm89`, `fn-swift1.5-huihui-quad-32g-sm89` | `27b-tp4-sm89-32g`, `27b-tp2-sm89-32g`, `27b-tp1-sm89-32g` | `svl-quad-sm89-32g` |
| 2 x RTX 4090 48 GB (modified) | `fn-2x48g-pp2-sm89` | `fn-swift1.5-2x48g-pp2-sm89`, `fn-swift1.5-huihui-2x48g-pp2-sm89` | `27b-tp2-sm89-48g`, `27b-tp1-sm89-48g` | refused: no two-card Super VL layout |
| 4 x RTX 4090 48 GB (modified) | `fn-4x48g-tp2pp2-sm89` | `fn-swift1.5-4x48g-tp2pp2-sm89`, `fn-swift1.5-huihui-4x48g-tp2pp2-sm89` | `27b-tp4-sm89-48g`, `27b-tp2-sm89-48g`, `27b-tp1-sm89-48g` | `svl-quad-sm89-48g` |

Sizing rules. Flash-Next uses its existing rows. The Swift and Huihui rows clone the vanilla row's layout, partition and per-card KV pin (fp8 KV only; no bf16 variant is offered for the checkpoint profiles) with P2P-off exports; they were previously refused on any host without P2P. The 27B and Super VL rows keep the 3090 rows' **per-card KV pins unchanged**, so their pools are the 3090 pools by construction; the extra VRAM of the 32 / 48 GB classes is headroom (image peak, transients), not a larger pool, and the 8 MP image floor is never lowered. The 27B series patches and the FlashInfer 32-row verify tile are sm_86 only, so sm_89 takes the stock paths, whose numerics are not parity-qualified; Super VL ran on cloud-hosted 4 x RTX 4090 24 GB without P2P (2026-10-18 and 2026-10-20, in its 262K mode), and the 32 / 48 GB Ada rows have not run. The 27B and Super VL launchers take `P2P=on\|off`; `merlin serve` and preflight set it.

## Complete profile table

This block is a verbatim registry export. Every `~` pool or fit is an **untested estimate**, including octo, no-P2P and the Swift / Huihui rows other than the two quads. Matching the registry is a consistency check, not independent measurement evidence. No retained source record establishes the non-reference Flash-Next pins or the predicted capacities on those platforms.

Generated from `tools/merlin/profiles/flash-next.json` by `python tools/merlin/preflight.py --emit matrix`.

<!-- BEGIN matrix (generated: tools/merlin/preflight.py --emit matrix) -->
| Profile | Cards | Arch | P2P | Layout | seqs | KV | Pool (tokens) | 262K / 131K | Status |
|---|---|---|---|---|---|---|---|---|---|
| `fn-quad-3090` | 4 x 24 GB | sm_86 | on | TP2xPP2+EP | 8 | fp8 | 924,993 | 3 / 6 | tested on the reference host |
| `fn-quad-3090 kv-bf16` | 4 x 24 GB | sm_86 | on | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-quad-3090-nop2p` | 4 x 24 GB | sm_86 | off | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | untested / community-validated |
| `fn-quad-3090-nop2p kv-bf16` | 4 x 24 GB | sm_86 | off | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-quad-3090-nop2p-pp4` | 4 x 24 GB | sm_86 | off | TP1xPP4 | 32 | fp8 | 906,581 | 3 / 6 | tested on the reference host with P2P off (2026-10-19) |
| `fn-octo-3090-tp2pp4` | 8 x 24 GB | sm_86 | any | TP2xPP4+EP | 32 | fp8 | ~4,656,696 | ~17 / 31 | untested / community-validated |
| `fn-octo-3090-tp4pp2` | 8 x 24 GB | sm_86 | any | TP4xPP2+EP | 20 | fp8 | ~2,962,227 | ~11 / 20 | untested / community-validated |
| `fn-2x48g-pp2-sm86` | 2 x 48 GB | sm_86 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-2x48g-pp2-sm86 kv-bf16` | 2 x 48 GB | sm_86 | any | TP1xPP2 | 4 | bf16 | ~726,513 | ~2 / 4 | untested / community-validated |
| `fn-4x48g-tp2pp2-sm86` | 4 x 48 GB | sm_86 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-4x48g-tp2pp2-sm86 kv-bf16` | 4 x 48 GB | sm_86 | any | TP2xPP2+EP | 8 | bf16 | ~3,580,138 | ~13 / 24 | untested / community-validated |
| `fn-quad-24g-sm89` | 4 x 24 GB | sm_89 | any | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | 🟡 testing: full-model runs passed on cloud-hosted hardware; kernel parity qualification still to run |
| `fn-quad-24g-sm89 kv-bf16` | 4 x 24 GB | sm_89 | any | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-quad-24g-sm89-pp4` | 4 x 24 GB | sm_89 | off | TP1xPP4 | 32 | fp8 | ~506,083 | ~1 / 3 | untested / community-validated |
| `fn-quad-24g-sm80` | 4 x 24 GB | sm_80 | any | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-2x48g-pp2-sm89` | 2 x 48 GB | sm_89 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-2x48g-pp2-sm89 kv-bf16` | 2 x 48 GB | sm_89 | any | TP1xPP2 | 4 | bf16 | ~726,513 | ~2 / 4 | untested / community-validated |
| `fn-4x48g-tp2pp2-sm89` | 4 x 48 GB | sm_89 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-4x48g-tp2pp2-sm89 kv-bf16` | 4 x 48 GB | sm_89 | any | TP2xPP2+EP | 8 | bf16 | ~3,580,138 | ~13 / 24 | untested / community-validated |
| `fn-quad-32g-sm89` | 4 x 32 GB | sm_89 | any | TP2xPP2+EP | 19 | fp8 | ~2,868,604 | ~10 / 19 | untested / community-validated |
| `fn-quad-32g-sm89 kv-bf16` | 4 x 32 GB | sm_89 | any | TP2xPP2+EP | 8 | bf16 | ~1,516,690 | ~5 / 10 | untested / community-validated |
| `fn-sm80-80g-x1` | 1 x 80 GB | sm_80 | any | TP1xPP1 | 2 | bf16 | ~344,532 | ~1 / 2 | untested / community-validated |
| `fn-sm80-80g-x2-tp2` | 2 x 80 GB | sm_80 | on | TP2xPP1+EP | 20 | bf16 | ~2,980,951 | ~11 / 20 | untested / community-validated |
| `fn-sm80-80g-x2-pp2` | 2 x 80 GB | sm_80 | off | TP1xPP2 | 19 | bf16 | ~2,853,624 | ~10 / 19 | untested / community-validated |
| `fn-sm80-80g-x4-tp2pp2` | 4 x 80 GB | sm_80 | any | TP2xPP2+EP | 32 | bf16 | ~7,703,288 | ~28 / 52 | untested / community-validated |
| `fn-sm80-64g-x2-pp2` | 2 x 64 GB | sm_80 | any | TP1xPP2 | 12 | bf16 | ~1,790,069 | ~6 / 12 | 🟡 testing: full-model runs passed on cloud-hosted hardware; kernel parity qualification still to run |
| `fn-sm80-32g-x4-pp4` | 4 x 32 GB | sm_80 | any | TP1xPP4 | 8 | bf16 | ~1,219,697 | ~4 / 8 | untested / community-validated |
| `fn-sm80-32g-x3-pp3` | 3 x 32 GB | sm_80 | any | TP1xPP3 | 3 | bf16 | ~469,674 | ~1 / 3 | untested / community-validated |
| `fn-sm80-40g-x4-pp4` | 4 x 40 GB | sm_80 | any | TP1xPP4 | 14 | bf16 | ~2,122,638 | ~7 / 14 | untested / community-validated |
| `fn-sm80-64g-x4-pp4` | 4 x 64 GB | sm_80 | any | TP1xPP4 | 8 | bf16 | 4,638,492 | 17 / 31 | 🟡 testing: full-model runs passed on cloud-hosted hardware; kernel parity qualification still to run |
| `fn-swift1.5-quad-3090` | 4 x 24 GB | sm_86 | on | TP2xPP2+EP | 8 | fp8 | 924,993 | 3 / 6 | tested on the reference host |
| `fn-swift1.5-huihui-quad-3090` | 4 x 24 GB | sm_86 | on | TP2xPP2+EP | 8 | fp8 | 924,993 | 3 / 6 | tested on the reference host |
| `fn-swift1.5-quad-3090-nop2p` | 4 x 24 GB | sm_86 | off | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | untested / community-validated |
| `fn-swift1.5-quad-24g-sm89` | 4 x 24 GB | sm_89 | any | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | 🟡 testing: full-model runs passed on cloud-hosted hardware; kernel parity qualification still to run |
| `fn-swift1.5-quad-32g-sm89` | 4 x 32 GB | sm_89 | any | TP2xPP2+EP | 19 | fp8 | ~2,868,604 | ~10 / 19 | untested / community-validated |
| `fn-swift1.5-2x48g-pp2-sm89` | 2 x 48 GB | sm_89 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-swift1.5-4x48g-tp2pp2-sm89` | 4 x 48 GB | sm_89 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-swift1.5-huihui-quad-3090-nop2p` | 4 x 24 GB | sm_86 | off | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | untested / community-validated |
| `fn-swift1.5-huihui-quad-24g-sm89` | 4 x 24 GB | sm_89 | any | TP2xPP2+EP | 8 | fp8 | ~924,993 | ~3 / 6 | untested / community-validated |
| `fn-swift1.5-huihui-quad-32g-sm89` | 4 x 32 GB | sm_89 | any | TP2xPP2+EP | 19 | fp8 | ~2,868,604 | ~10 / 19 | untested / community-validated |
| `fn-swift1.5-huihui-2x48g-pp2-sm89` | 2 x 48 GB | sm_89 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-swift1.5-huihui-4x48g-tp2pp2-sm89` | 4 x 48 GB | sm_89 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-swift1.5-2x48g-pp2-sm86` | 2 x 48 GB | sm_86 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-swift1.5-4x48g-tp2pp2-sm86` | 4 x 48 GB | sm_86 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-swift1.5-quad-24g-sm80` | 4 x 24 GB | sm_80 | any | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-swift1.5-sm80-80g-x1` | 1 x 80 GB | sm_80 | any | TP1xPP1 | 2 | bf16 | ~344,532 | ~1 / 2 | untested / community-validated |
| `fn-swift1.5-sm80-80g-x2-tp2` | 2 x 80 GB | sm_80 | on | TP2xPP1+EP | 20 | bf16 | ~2,980,951 | ~11 / 20 | untested / community-validated |
| `fn-swift1.5-sm80-80g-x2-pp2` | 2 x 80 GB | sm_80 | off | TP1xPP2 | 19 | bf16 | ~2,853,624 | ~10 / 19 | untested / community-validated |
| `fn-swift1.5-sm80-80g-x4-tp2pp2` | 4 x 80 GB | sm_80 | any | TP2xPP2+EP | 32 | bf16 | ~7,703,288 | ~28 / 52 | untested / community-validated |
| `fn-swift1.5-sm80-64g-x2-pp2` | 2 x 64 GB | sm_80 | any | TP1xPP2 | 12 | bf16 | ~1,790,069 | ~6 / 12 | 🟡 testing: full-model runs passed on cloud-hosted hardware; kernel parity qualification still to run |
| `fn-swift1.5-sm80-32g-x4-pp4` | 4 x 32 GB | sm_80 | any | TP1xPP4 | 8 | bf16 | ~1,219,697 | ~4 / 8 | untested / community-validated |
| `fn-swift1.5-sm80-32g-x3-pp3` | 3 x 32 GB | sm_80 | any | TP1xPP3 | 3 | bf16 | ~469,674 | ~1 / 3 | untested / community-validated |
| `fn-swift1.5-sm80-40g-x4-pp4` | 4 x 40 GB | sm_80 | any | TP1xPP4 | 14 | bf16 | ~2,122,638 | ~7 / 14 | untested / community-validated |
| `fn-swift1.5-huihui-2x48g-pp2-sm86` | 2 x 48 GB | sm_86 | any | TP1xPP2 | 9 | fp8 | ~1,419,322 | ~5 / 9 | untested / community-validated |
| `fn-swift1.5-huihui-4x48g-tp2pp2-sm86` | 4 x 48 GB | sm_86 | any | TP2xPP2+EP | 32 | fp8 | ~6,767,060 | ~25 / 46 | untested / community-validated |
| `fn-swift1.5-huihui-quad-24g-sm80` | 4 x 24 GB | sm_80 | any | TP2xPP2+EP | 3 | bf16 | ~486,838 | ~1 / 3 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-80g-x1` | 1 x 80 GB | sm_80 | any | TP1xPP1 | 2 | bf16 | ~344,532 | ~1 / 2 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-80g-x2-tp2` | 2 x 80 GB | sm_80 | on | TP2xPP1+EP | 20 | bf16 | ~2,980,951 | ~11 / 20 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-80g-x2-pp2` | 2 x 80 GB | sm_80 | off | TP1xPP2 | 19 | bf16 | ~2,853,624 | ~10 / 19 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-80g-x4-tp2pp2` | 4 x 80 GB | sm_80 | any | TP2xPP2+EP | 32 | bf16 | ~7,703,288 | ~28 / 52 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-64g-x2-pp2` | 2 x 64 GB | sm_80 | any | TP1xPP2 | 12 | bf16 | ~1,790,069 | ~6 / 12 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-32g-x4-pp4` | 4 x 32 GB | sm_80 | any | TP1xPP4 | 8 | bf16 | ~1,219,697 | ~4 / 8 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-32g-x3-pp3` | 3 x 32 GB | sm_80 | any | TP1xPP3 | 3 | bf16 | ~469,674 | ~1 / 3 | untested / community-validated |
| `fn-swift1.5-huihui-sm80-40g-x4-pp4` | 4 x 40 GB | sm_80 | any | TP1xPP4 | 14 | bf16 | ~2,122,638 | ~7 / 14 | untested / community-validated |
<!-- END matrix -->

The Super VL row is the reference entry in `tools/merlin/profiles/nemotron-super-vl.json`; its launcher is [`models/nemotron-super-vl/serve.sh`](../../models/nemotron-super-vl/serve.sh).
The Ornith 1.5 rows are in `tools/merlin/profiles/ornith.json`; [`models/ornith1.5/serve.sh`](../../models/ornith1.5/serve.sh) pins the P2P row, and `merlin serve ornith1.5 <dir>` picks the no-P2P row on a host without peer-to-peer.
The 27B rows are reference entries in `tools/merlin/profiles/qwen38-27b.json`. The launchers they describe are
[`models/qwen38-27b/serve.sh`](../../models/qwen38-27b/serve.sh) (TP2) and [`models/qwen38-27b/serve-tp4.sh`](../../models/qwen38-27b/serve-tp4.sh) (TP4);
the single-card profile is [`models/qwen38-27b/serve-1card.sh`](../../models/qwen38-27b/serve-1card.sh) (tested 2026-10-21: pool 271,506, 4 MP images).

## Where the numbers come from

| Figures | Provenance | What is established |
|---|---|---|
| Flash-Next reference pool 924,993; pin 4.1e9 bytes; block 4,096; nominal 3 x 262K / 6 x 131K fit | Historical v3.0.0 qualification, 2026-10-10, plus the 2026-10-15 claims review confirming the same pool and pin on vLLM 0.32 | The reference pool is measured. The rounded context fits are capacity summaries, not a claim that all requests use exactly those prompt lengths. |
| Flash-Next design concurrency; 27B TP4 design concurrency | Design-point concurrency run, 2026-10-15, a pre-release engine build (vLLM 0.32), 4 x RTX 3090 at 220 W | Flash-Next: six concurrent prompts of 126,976 tokens with 512 generated tokens each, zero preemptions. 27B TP4: fourteen at that depth with 1,536 generated tokens each, zero preemptions, on the RecoverSSM-off line that the shipped TP4 geometry replaced. No 27B TP2 design cell ran. |
| 27B reference pool 826,689 (TP2) and its nominal fit; TP4 2,012,105 | TP2: 27B v1.3.0 evidence index and benchmark results, 2026-10-08. TP4: the shipped TP4 geometry (RecoverSSM on, block 1024, chunk 4096, pin 17,923,833,856 B) | TP2: historical pool evidence, not a fresh pool measurement in the 2026-10-15 design-point run. TP4: logged at boot in the geometry run with every correctness check passed; the 2026-10-20 release bench card then measured the pool and decode 252.4 / 235.4 / 224.0 tok/s at 32K / 131K / 259K (thinking on), after the pre-release bench run (2026-10-17: 27B pair PASS, decode 198.7 / 172.6 / 157.7 tok/s at 32K / 131K / 258K; ThinkingCap check PASS, 32/32 checks). The TP1 pool (271,506) and fit were measured at the single-card release check on the 0.32 engine, 2026-10-21. |
| sm_80 smoke run: 4 x CMP 170HX 64 GB (a second host, driver 610.43.03, PCIe Gen2 x16 on all four cards, P2P off), 2026-10-18 | Smoke-grade runs: boot, needles at 32K-262K, design-point concurrency, GSM8K-50, image, a decode/TTFT card, idle power. Not the release bench routine | Measured pools: `fn-sm80-64g-x2-pp2` 1,790,069; `fn-sm80-40g-x4-pp4` (booted on 64 GB cards) 2,122,638; `27b-tp1/-tp2/-tp4-sm80-64g` 1,425,876 / 3,136,365 / 6,640,647; `svl-quad-sm80-64g` 22,777,400. All needles exact. Flash-Next and Swift 1.5 then died at the shipped pin on the first prefix-cache hits (GSM8K at concurrency 8): the copy-on-write block copy allocated n_copies x the per-rank block (1.30 / 1.58 GiB). The copy is now a row copy with no temporary. A second set of cloud-hosted cards (2026-10-19) passed both PP2 rows at that pin with the row copy (GSM8K-50 c8 0.98 three times, every prefix hit, 0 OOM), the 4 x 64 GB PP4 row and the 27B TP1 / TP2 / TP4 layouts; the rows that passed are 🟡 testing. |
| All other pools, fits, pins and block counts, including the octo values below | `tools/merlin/profiles/flash-next.json`, fields `serve` and `capacity` | **Untested estimates only.** The original capacity probe and its inputs/results are not distributed here, and no matching independent validation record was found. |
| Layouts, GPU classes, sequence caps, capture sizes, MTP settings, image-cap default and selector thresholds | Checked-in profile registries, `registry.py`, `preflight.py` and `serve_cli.py` | Configuration or policy; these numbers are not measured hardware guarantees. |

KV dtype follows the engine check (`_validated_qsa_fp8_dtype`, `vllm/models/qwen4_exp/nvidia/qsa.py`): fp8_e4m3 KV is admitted on sm_80/sm_86 (Merlin byte decoder) and sm_89+ (upstream native cast). **BF16 KV remains the default on every sm_80 Flash-Next, Swift and Huihui row.** Set `KV_DTYPE=fp8` for its launcher or `merlin serve` (also read by preflight) to opt into fp8_e4m3 with that row's separate `capacity_fp8_opt_in` engine-fit estimate. FP8 status is **⚫ untested: a profile ships, not yet run by us, expected to work**; its pool figures are not measured sm_80 results. The knob accepts only `fp8`; unset it for the profile default, and other values are refused. It is Flash-Next-only and a no-op on the already-FP8 sm_86/sm_89 rows. On sm_80 it prints: "sm_80 FP8 KV: untested by us (no sm_80 run yet); BF16 is the default". The sm_86 and sm_89 rows continue to default to fp8 KV with bf16 selectable (a generated `-kv-bf16` launcher, `--emit launcher --kv bf16`, about half the pool). A bf16 variant's pool is an untested estimate; the tested quad row stays fp8 and byte-identical to `models/flash-next/serve.sh`.

The evidence descriptions identify the dated qualification, design-point, image-floor and identity record summaries used for this page; raw logs are not distributed with these docs. Configuration values can be checked in the named source files. A registry entry alone is not independent numeric provenance.

The reference pool can calibrate a capacity model, but does not validate another layout's pin, overhead or fit. The undocumented probe's detailed weight/overhead breakdown, alternative partitions and alternative pins have therefore been removed from this page.

**Image evidence has a separate scope.** The image-floor run on 2026-10-15 used the 8,388,608-pixel cap and recorded stage-0 minimum free memory of 902 MiB for Flash-Next and Swift1.5, and 690 MiB for 27B TP4 (on the retired RecoverSSM-off pin; at the shipped chunk 4096 with RecoverSSM on, the 2026-10-20 bench card measured 862 MiB, 2 cards 886 MiB). The initial 27B TP2 window reached 526 MiB, but the later stress run, before the vision-peak option existed, reached 506 MiB and failed its 524 MiB floor despite completing without an OOM (a pre-fix FAIL, superseded by the result below). With the vision-peak option on (`VLLM_MERLIN_VISION_PEAK=1`, the TP2 launcher default), the same 50-image stress set on the release engine reached 806 MiB on both cards, above the floor, with outputs identical to the option off. The launcher qualification (2026-10-16) measured the generated Swift launcher at 544 MiB stage-0 minimum free after an 8K prefill (20 MiB above the floor) and the generated Huihui launcher at 902 MiB. These are specific run windows, not a universal free-memory guarantee or evidence for every allowed image-count combination.

The merged-engine identity run on 2026-10-15 compared the merged engine with the 0.32 reference build: tokens and logprobs matched at temperature zero for 28 documents plus four controls on each of four served configurations. The release-candidate identity run agreed. Identity does not re-measure memory, throughput or another hardware platform. The separate 27B verify-tile run changes arithmetic when enabled and must not be conflated with this identity result.

**FP8 KV by architecture.** The Flash-Next QSA reader admits FP8 E4M3 KV on sm_80 and sm_86 (the same Merlin byte decoder) and on sm_89 or newer (the native cast), via `_validated_qsa_fp8_dtype` in `vllm/models/qwen4_exp/nvidia/qsa.py` and `qsa_e4m3_uses_native_cast` in `nvidia/ops/qsa.py`. `registry.FP8_KV_ARCHS` records engine admission, not the default: every sm_80 Flash-Next row stays BF16. `KV_DTYPE=fp8` selects the FP8 opt-in capacity and the E4M3-only `VLLM_QSA_KV_SCALES` sidecar. Its pools are engine-fit estimates, not measured or simply doubled BF16 pools: the compressed indexer and ring state keep their existing dtypes. The 4 x 64 GB PP4 row retains its measured BF16 pool of 4,638,492 tokens; its FP8 opt-in pool ~9,007,559 (~34 x 262K / ~61 x 131K, admission still capped at 8) is an untested engine-fit estimate. Historical BF16 evidence does not qualify FP8. All sm_80 Flash-Next FP8 opt-ins are ⚫ untested (a profile ships, not yet run by us, expected to work). No sm_80 FP8 hardware validation is claimed. The 27B and Super VL FlashInfer paths have no such engine guard, so their sm_80 rows are sized with FP8 KV like every other row. FP8 KV has now booted on sm_80 for both (4 x CMP 170HX smoke, 2026-10-18): 27B TP1 / TP2 / TP4 and Super VL TP2xPP2 ran with fp8_e4m3 KV through FlashInfer, every needle exact, GSM8K-50 48/50 on each. The 27B sm_80 TP2/TP4 MTP verify runs PIECEWISE (the series patches are limited to sm_86). These 64 GB rows are 🟡 testing: kernel parity qualification is still to run. The selector can return FP8 for sm_89 with an uncalibrated-pool warning; the BF16 capacity figures do not predict that FP8 pool.

## Flash-Next on 8 x RTX 3090 (octo)

On an 8-card RTX 3090 host with P2P, `merlin serve flash-next` prefers the tested `fn-quad-3090` row on four
cards and notes that four matching cards stay idle. To select the eight-card layout, use
`merlin serve flash-next /path/to/checkpoint --profile fn-octo-3090-tp2pp4` (untested / community-validated).

The registry default is **TP2xPP4+EP**, partition 12,12,12,12, admission cap 32, FP8 KV and capture sizes through 128 decode tokens. Its **11.25 GiB pin, 4,656,696-token pool and 17 x 262K / 31 x 131K fits are untested estimates**, not an accepted memory budget. The TP4xPP2 alternative has admission cap 20 and the same evidence limitation.

| Registry profile | KV pin per card (untested estimate) | Pool (untested estimate) | 262K / 131K fit (untested estimate) |
|---|---|---|---|
| `fn-octo-3090-tp2pp4` | 11.25 GiB | 4,656,696 | 17 / 31 |
| `fn-octo-3090-tp4pp2` | 12.19 GiB | 2,962,227 | 11 / 20 |

The model has two KV heads, so increasing TP can duplicate KV heads rather than reduce per-rank KV storage. Additional pipeline stages can reduce layers per rank. This motivates the capacity-oriented default and the latency alternative; no measured eight-card comparison is claimed.

There is no qualifying eight-card Merlin run in the reviewed evidence. Earlier community throughput ranges and pool comparisons have been removed because their underlying reports were not established by the retained records. The extra graph/sampler overhead at the higher admission cap is unmeasured; no numeric allowance is claimed.

Before either profile is validated, report a pool-only boot, peak memory, the chosen partition and pin, and the decode, needle and smoke checks below. Admission caps are configuration, not a promise of resident capacity.

## Other platforms

- **sm_86 with more VRAM (RTX A6000, A40).** The profiles reuse the reference FP8 path with different layouts or pins. Their capacities remain untested registry estimates. An A40 reports ~45 GiB against the 48 GiB class, so preflight lowers the pin by ~3 GiB (same mechanism as the L40S). The 4 x 48 GB sm_86 profile now has a no-P2P (`P2P=off`) mode like its sm_89 sibling, so a quad of A40s or A6000s without peer access is not launched with P2P on. No official cloud 2 x 48 GB or 4 x 48 GB sm_86 shape was found (Lambda's historical A6000 instances had NVLink pairs only).
- **22.5 GiB cards (A10, A10G, L4) and 24 GiB cards that are not 3090s.** Matching never reads the card name: preflight takes compute capability (name regex only as a fallback, which now recognises `A10G`), the VRAM class (24 GiB +-7 %: AWS lists 22 GiB for A10G and L4, and `nvidia-smi` samples of 23,028 MiB (A10 / A10G) and 23,034 MiB (L4), illustrative and not independently verified, are inside the class; the preflight simulation uses those values) and the P2P state. AWS does not document P2P; our pins were set on 24,576 MiB cards, so the pin is lowered by the reported shortfall (`kv_bytes_for_vram`, the L40S mechanism) and preflight prints the resulting pool as an estimate scaled linearly from the profile sizing, never a 3090 figure. A 22.5 GiB card is "sized for the hardware", not squeezed: the pin is card VRAM - binding-stage weights - 2.75 GiB overhead, and a card whose adjusted pool cannot hold the tokens of one 262K session is refused (the test is the adjusted pool, not a scaled session count; the scaled fits under-report). A quad of A10G holds one 262K session. Cloud shapes: AWS g5.12xlarge / g5.24xlarge (4 x A10G, 192 / 384 GiB host RAM), g5.48xlarge (8 x A10G, 768 GiB), g6.12xlarge (4 x L4, 192 GiB), GCP g2-standard-48 (4 x L4, PCIe Gen4, no NVLink). Azure NV A10 v5 shapes are fractional GPUs except NV36 (1 x A10) and NV72 (2 x A10), which do not fit Flash-Next. Cloud GPUs sit behind a hypervisor, so expect the VM warning and run `nvidia-smi topo -p2p r`; with P2P off the no-P2P profiles are chosen.
- **The tested label is name-gated.** Every row marked tested on the reference host (`fn-quad-3090`, `27b-tp1-3090`, `27b-tp2-3090`, `27b-tp4-3090`, the Swift1.5 and Huihui quads, `svl-quad-3090`, `ornith-quad-3090`, `ornith-quad-3090-nop2p`) carries `match.tested_gpu_names: ["RTX 3090$"]`. Any other card that resolves to the row (RTX A5000, RTX 3090 Ti, 150 W A10 at 24 GB, ...) is reported **untested / community-validated** with a `tested-hw` warning. The 🟡 testing label is gated the same way by `match.testing_gpu_names` (`RTX 4090$` or `CMP 170HX`): an L4 or A30 that resolves to such a row is reported untested with a `testing-hw` warning.
- **Ampere sm_80 (A100, A30, CMP 170HX).** Build for sm_80; BF16 KV stays the default and `WARM=0`. `KV_DTYPE=fp8` opts into the byte decoder with the separate engine-fit capacity estimate, labelled ⚫ untested (a profile ships, not yet run by us, expected to work). Community findings that inform these rows, configuration only (no community throughput is quoted here): DUFRelic's recipe (github.com/DUFRelic/qwen38-flash-next-recipe, vLLM 0.30.0, 2 x 170HX, PP2) runs BF16 KV because the pinned 0.30.0 QSA path rejects FP8 KV; a community 2 x CMP 170HX recipe reports FP8 KV gave +36 % capacity but 6-11 % slower decode, again on BF16 by default; a community CMP overlay lists FP8 in its README, which we could not reconcile with the others. This engine admits sm_80 FP8 KV through the sm_86 byte-decoder path. Community evidence is a decoder port from the club-3090 community (discussion #1735); it is not a Merlin hardware-validation result. On a stock 170HX unlock P2P is off (`canAccessPeer` false; DUFRelic needed patched BAR1 drivers to turn it on), so those rows default to P2P off. DUFRelic and that overlay serve a community repack of the Merlin checkpoint (INT4 PLE, FP8 KV).
  - **Our 4 x 170HX smoke run (2026-10-18; the measured pools are in "Where the numbers come from").**
    - Host: every card reported PCIe Gen2 x16, `nvidia-smi topo -p2p r` reported GNS on every pair, and the default power limit was 250 W.
    - Flash-Next prefill: at 262K it sat at the software power cap (throttle reason 0x4).
    - Idle: the cards never left P0, including a card holding no engine. An engine-held idle card read ~36-44 W against ~25 W at rest, so the P8 idle path is not observable on this SKU.
    - The plain release install (`install-env.sh`, `TORCH_CUDA_ARCH_LIST=8.0`) verified in 27 minutes.
    - Layout: `merlin serve flash-next` without `GPUS` picks the 4-card `fn-sm80-64g-x4-pp4` (🟡 testing) on 4 x 64 GB; `merlin doctor` reports the same choice. Next to a class with a tested row (4 x CMP 170HX + 4 x RTX 3090) the tested class is picked first. On 8 cards the default is one PP4 server on GPUs 0-3 plus a hint for GPUs 4-7; `--instances 2` starts two (host RAM permitting: ~69 GiB each). The 2-card `fn-sm80-64g-x2-pp2` stays as the alternative for a pair.
    - PP4: on all four 64 GB cards it gave ~70 % faster long-context prefill than the 2-card PP2 row, with GSM8K-50 at 49/50; the row's own runs on cloud-hosted cards on 2026-10-19 are behind its 🟡 testing label.

  These profiles still need full qualification, the release bench routine, on the actual hardware.
- **Consumer Ada has no P2P by default.** NVIDIA's stock driver exposes no peer access on GeForce Ada (tinygrad/open-gpu-kernel-modules is the patched module that adds it), and a community 2 x RTX 4090 run of the 27B family without P2P ran slower with TP2 than on one card (HyperQwen's multi-GPU notes). So the 4090 rows default to `P2P=off` and preflight picks `on` only when the driver reports it. On a no-P2P host the 27B default stays the largest TP (the largest pool); `merlin serve` prints a NOTE with the measured aggregate (4 x CMP 170HX 64 GB without P2P, 8 agents x 16K: TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371; 4 x 4090 without P2P showed the same order for prefill, TP1 ~970-1,435 > TP2 ~909-1,096 > TP4 ~830-964 tok/s) and the `merlin serve 27b <dir> --tp 1 --instances N` command for that host.
- **Lovelace sm_89 (RTX 4090, modified RTX 4080 Super, L40S, RTX 6000 Ada, L4).** Build for sm_89 and use `WARM=0`; the profiles default to fp8 KV with bf16 selectable. The native E4M3 path has run on cloud-hosted 4 x RTX 4090 24 GB (2026-10-18: six of seven lanes passed, Flash-Next on fp8 KV with the 924,993-token pool; 2026-10-20: the release wheel passed all three lanes), so that row is 🟡 testing; kernel parity qualification is still to run, and the other Ada cards and the bf16 variant still need hardware calibration. Feed reported VRAM into preflight so a shortfall lowers the pin; a GPU product label is not a pool measurement. There is no persisted Triton autotune cache on sm_89 (`WARM` is sm_86 only), but the GDN chunked-prefill kernels do not autotune on any compute-capability-8.x card: on sm_80, sm_86 and sm_89 alike they launch one fixed configuration, the one the sm_86 reference caches ran (`autotune_off_sm8x` / `sm8x_launch` in `vllm/third_party/flash_linear_attention/ops/utils.py`, `is_nvidia_sm8x` = major version 8). The first boot therefore compiles those kernels without an autotune search; the q/k L2-norm kernels still autotune live. The fixed configurations were chosen on sm_86, not tuned for Ada: a pool-only boot plus a decode check at 4K and 131K is the validation. On WSL2 the driver's free-memory reading is reported unreliable by the community; preflight warns.
- **No P2P.** NCCL transports communication through host memory when peer access is unavailable. The unchanged-layout and PP-heavy alternatives remain untested for this release. The PP4 registry pool is an estimate here; no qualifying source for the previously claimed four-card smoke and partition result was found. A four-card pipeline test would not establish the eight-card memory budget.

## Host requirements

Same for every Flash-Next row unless noted. The preflight checks all of them. The Flash-Next reference is [`models/flash-next/docs/reference.md`](../../models/flash-next/docs/reference.md).

| Requirement | Value | Source |
|---|---|---|
| Host RAM | **96 GiB (98,304 MiB)** allocation. The FP8 PLE table occupies about 47.7 GiB of host memory; pinned embedding tables are separate. The roughly 69 GiB resident floor was measured on v2.0.1 and has not been re-measured on vLLM 0.32. Additional-worker RAM is estimated by preflight, not measured. | Historical memory and launcher reviews, 2026-09-25; claims review, 2026-10-15 |
| `/dev/shm` | Preflight warns below 1 GiB; container recipes allocate 8 GiB. This is a sizing policy, not a measured boot peak. | Flash-Next reference and preflight |
| Driver | Preflight's 580-series check matches the CUDA 13.x floor. Reference host: NVIDIA driver 610.43.02 (reports CUDA 13.3). The release lock selects a matched CUDA 13.0 toolkit (nvcc, crt, nvvm, nvJitLink 13.0.88, cccl 13.0.85, runtime 13.0.96, the pair torch 2.13.0 cu130 ships with); the release wheel is served on it by every recipe (2026-10-20). Other CUDA 13.x drivers (R580 or later) are expected to work through CUDA minor-version compatibility, but are untested. See the [driver requirements](../../release/install-0.32.md). | Reference-host qualification, 2026-10-15; current lock; preflight |
| GPU state | Headless, with no other CUDA process. Use the image-run minima above rather than a generic per-card headroom claim. | Image-floor run, 2026-10-15 |
| glibc | **2.38 or newer** (Ubuntu 24.04, Debian 13, RHEL 10 or newer): the release wheel's `_C_stable_libtorch.abi3.so` needs GLIBC_2.38. | Release wheel symbol requirements |
| Host type | Bare metal or LXC; VM passthrough and WSL2 receive a preflight warning ([WSL2](#wsl2)). | preflight policy |
| BAR1 (P2P only) | Preflight vetoes P2P below 1 GiB per card. Check `nvidia-smi -q -d MEMORY` and the effective P2P report. | preflight policy |
| Persistence before a power cap | Enable persistence before setting the cap; confirm the resulting state. | preflight policy |
| PCIe / power | Reference configuration: Gen4 x16 links, 220 W cap. Other links and caps need measurement. | Historical hardware record; design-point run, 2026-10-15 |
| Python / toolchain | Use the Flash-Next recipe's documented interpreter and build tools, with a CUDA toolkit compatible with the driver. | Flash-Next reference |

## WSL2

Windows runs the engine only under WSL2, and no checkpoint has run there (⚫ untested; preflight warns and never
reports a tested row on WSL2). If you try it:

- **Pinned memory.** vLLM turns pinned host memory off on WSL2 unless `VLLM_WSL2_ENABLE_PIN_MEMORY=1` is set, on a
  WSL2 kernel 4.19.121 or newer (`wsl --update`). The host-resident embedding tables need it. Other buffers fall back to
  device memory without it, which costs VRAM. The 27B, Super VL and Ornith launchers set it on WSL2 (export it yourself
  for a direct `vllm serve`). Flash-Next's launcher leaves it to you, because Flash-Next can pin tens of GiB. The
  setting guarantees neither capacity nor a working mapping: NVIDIA notes that pinned-memory availability under WSL is
  limited ([CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)), and vLLM warns that over-using it
  can make Windows unresponsive.
- **WSL memory limit.** WSL2 gets 50 % of Windows RAM by default. Raise `memory=` in `%UserProfile%\.wslconfig`
  ([WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config)), then run `wsl --shutdown`. Host RAM
  below the requirement can end the model load in the kernel's OOM killer.
- **Peer-to-peer.** Peer access is usually absent under WSL2 (unverified on our side), and preflight treats an
  unreadable P2P probe as off, so `merlin serve` picks the no-P2P row. With a serve script, set `P2P=off`.
- **The display card.** Windows holds VRAM on the card that drives a display, and the profiles pin KV in bytes for an
  otherwise empty card. Serve on headless cards (`GPUS=`). For the 27B, `KV_BYTES=auto` is an unqualified attempt at a
  smaller pool. The launcher still asks for `--gpu-memory-utilization 0.96`, so a busy card may also need a lower value
  passed as an extra argument. Check the resulting pool before relying on it.
- **Driver and toolkit.** Install the NVIDIA driver on Windows only, never a Linux driver inside WSL. The 0.32
  environment carries its own CUDA 13.0 toolkit, so leave `CUDA_HOME` unset. An external toolkit must pass the
  release's toolkit check ([install-0.32.md](../../release/install-0.32.md)): use the versioned, driver-free
  `cuda-toolkit-13-0` package, not the unversioned `cuda-toolkit`, which tracks newer releases.
- **nvidia-smi.** It has a limited feature set under WSL, and persistence mode cannot be changed, so skip the
  `nvidia-smi -pm 1` step. Set a power limit from Windows where the card and driver support it.

Reports from WSL2 runs are welcome ([Reporting a run](#reporting-a-run-the-community-validation-checklist)).

## Optional host upgrade: GPU unlock (GeForce GA102)

Like P2P, this is an optional, platform-level change. It is not a Merlin component, and Merlin runs without it; the
3.0.1 bench numbers are measured with it (last paragraph).
[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) is a third-party host kernel module (GPL-2). It loads
before the NVIDIA driver binds the card and lifts the GeForce cap on FP16/BF16 tensor math with FP32 accumulate,
which the W4A16 GEMMs and attention use during prefill.

Measured on the reference host (4 x RTX 3090, 220 W cap, P2P on, 2026-10-25). Each arm is one boot, salted cold
prompts, single stream:

| Recipe | Prefill at 262K, stock to unlocked | Decode |
|---|---|---|
| Flash-Next TP2xPP2, MTP K=3 | about +16 % at every depth (TTFT 44.6 s to 38.3 s at 262K) | flat |
| Qwen3.8-27B TP2 | +28.6 % (TTFT 303.5 s to 235.8 s) | flat |

- 27B logprobs matched stock at tolerance 0, and GEMM outputs were bit-exact. P2P stayed intact. The unlocked boot logged
  0 corrected AER errors on all four Gen4 x16 links.
- Prefill is the lever; decode does not move.
- The gain is power-bound. Under GEMM load at a 215 W cap, one unlocked card held about 750 MHz (BF16 GEMM +22 % at free
  clocks): the gain is clock-limited at the power cap.
- DFlash2 (`--spec dflash2`) adds roughly another 10 % prefill on top, both stock and unlocked. The gain comes from its
  2048-token prefill chunk (the default is 1024), not from the drafter.

Before you install it:

- **Cold boot.** The unlock applies at a cold boot (power off, then on). A warm reboot or module reload does not
  change the state. Unloading the module does **not** re-lock a card: it stays unlocked until its next power-off.
- **Firmware stays local.** The module's build step generates NVIDIA firmware images that are for personal use only.
  Never redistribute them. Merlin does not ship or download them.
- **Version.** Upstream commit fc83321 broke RTX 3090 unlocks (upstream issues #20 and #23), and main was reverted to
  c274e74. Use a revision without that regression and check upstream's issue tracker first. This note is as of
  2026-10-24.
- **Pinned kernels.** The repository's `make install-deb` installs a DKMS package. On a host with a pinned kernel and a
  patched (P2P) open driver, pulling in `dkms` can rebuild or re-arm the stock driver tree. Build the `.ko` against
  the running kernel's headers instead.
- **Opt-in per card.** The module takes `gpu_unlock.targets=`. Unlock a single card first and compare it against a
  stock card before unlocking the set.

**Checking it.** `merlin doctor` prints one line per card and one for the module, for example:

```text
  GPU0 (0000:01:00.0): GPU unlock: on
  GPU unlock module gpu_unlock: loaded
```

The check covers the GeForce cards the module supports: Ampere GA102, GA103, GA104 and GA106 (RTX 3050 to 3090 Ti)
and Ada AD102, AD103, AD104, AD106 and AD107 (RTX 4060 to 4090, and 4090 D). It is tested on RTX 3090 only; the
other cards are untested ⚫ (expected to work), and upstream lists its Ada support as experimental. Turing cards are
not covered. Each card in the JSON report carries its chip `family` (for example `GA102` or `AD102`).

The card state is `on`, `off` or `unknown (<why>)`, in the text and the JSON report; `on` means the card's registers
hold the values the unlock module writes. A card that reads off while
the module is loaded prints `off (the unlock module is loaded; check `journalctl -k -b | grep gpu_unlock`)`: most
likely the unlock failed on this boot. The reasons for `unknown`, in the order they are checked:

- `not a Linux host`;
- `no PCI bus id from nvidia-smi`, `cannot read the PCI device id`;
- `<vvvv:dddd> is not a supported GeForce Ampere or Ada card`;
- `inside a container; check on the host`, `inside a VM; check on the host`, `inside WSL2; check on the host`,
  `cannot confirm bare metal`: bare metal is when `systemd-detect-virt` says `none`, and any other answer, or none,
  gives one of these;
- `needs root to read the card's registers`;
- `driver is <name>, not nvidia; registers not read`, `BAR0 is <size> bytes, too small for the feature registers;
  registers not read`;
- `register access refused: <error>`;
- `register read failed: timed out`, `... killed by <SIGNAME>`, `... exit code <n>`, `... malformed output` or
  `... <OS error>` (the read could not start);
- `register read returned an error pattern: <registers>` (the card answered with an error value, not register
  contents), `registers partly unlocked: <registers>`;
- `internal error: <msg>`.

So run it as root on the host itself. The module line reads `loaded`, `not loaded`, `unknown (not a Linux host)`,
`unknown (cannot read /proc/modules)` or `unknown (internal error: <msg>)`. The two lines are independent: a card stays unlocked until its next power-off even
after the module is unloaded, and a loaded module does not by itself mean the card was unlocked at a cold boot.
The check only reads the card's registers and never writes one. The read runs in a short-lived child process with a 5 s
timeout, so a hardware fault cannot crash `merlin doctor`. It runs on a live `merlin doctor` only, not with
`--dry-run`, `--simulate` or `--inventory`.

From 3.0.1 on, every Merlin bench number is measured with this unlock on (220 W cap) plus P2P, and credited as such
([README](../../README.md), bench basis). The 3.0.0 bench cards and earlier published numbers are stock-card numbers;
the unlock was first loaded on 2026-10-24, for the A/B above. Say in a run report whether the unlock was on (item 1 of
the checklist below).

## Running it

Run from the repository root on Linux with Bash, after building this checkout for sm_86, installing its serving environment in `venv/`, and making the Flash-Next recipe assets available under `models/flash-next/`. The scales sidecar must match the checkpoint and release; the launcher checks its presence. The draft head scores the full vocabulary (`DRAFT_HEAD=bf16|int8|int4`, default `bf16`); reduced draft vocabulary was removed in 3.0.1, and `REDUCED_DRAFT_VOCAB` or `DRAFT_VOCAB` set in the environment stops the launch (an empty value, or `REDUCED_DRAFT_VOCAB=0`, is accepted). A source checkout alone is insufficient.

```bash
test -f vllm/__init__.py                                  # the built tree (TREE defaults to this checkout)
test -x venv/bin/python && test -x venv/bin/vllm          # the serving venv (VENV defaults to <checkout>/venv)
test -f models/flash-next/scales/qsa_kv_scales_262k.json  # the scales sidecar (SCALES overrides it)
venv/bin/python tools/merlin/preflight.py

mkdir -p scripts
venv/bin/python tools/merlin/preflight.py --emit launcher --profile fn-octo-3090-tp2pp4 > scripts/serve-octo-3090-tp2pp4.sh
bash scripts/serve-octo-3090-tp2pp4.sh /path/to/Qwen3.8-Flash-Next-W4A16-Merlin
```

The generated file is the shim checked in as `tools/merlin/launchers/serve-octo-3090-tp2pp4.sh`, which can be run directly. A shim finds the checkout by walking up from its own directory to the one holding `tools/merlin/launch.py` (`MERLIN_HOME` overrides that); `TREE`, `VENV`, `CACHE_ROOT` and `SCALES` override the launcher's defaults. Invoke the redirected file with `bash`; redirection does not make it executable. To use the latency alternative, substitute `fn-octo-3090-tp4pp2` and its output filename in the generation and invocation commands. A test runs this generation and the shim from `scripts/`, but an end-to-end eight-card launch has not been tested. `MERLIN_PYTHON` names the interpreter a shim runs (default `python3`).

The generated launchers are shims for the one launcher (`tools/merlin/launch.py --profile <id>`), which builds the
Flash-Next recipe's environment and `vllm serve` flags from the profile and changes only the platform part (GPUs,
interconnect, layout, KV pin, concurrency, cudagraph capture sizes, `WARM` on non-sm_86). Since Merlin 3.0.0 every row
runs the recipe's build-product hash, toolchain, PLE-dtype, shadow, host-RAM and `/dev/shm` checks, and the recipe's
`topo_select.py` (`AUTO_TOPO=1`) on four-card P2P-on rows. [`models/flash-next/serve.sh`](../../models/flash-next/serve.sh)
pins the tested row.

## Reporting a run (the community validation checklist)

A row whose full-model runs passed on cloud-hosted cards is 🟡 **testing** until kernel parity qualification runs. A row becomes **tested** when someone runs it on v3 and reports all of:

1. hardware: cards, VRAM as `nvidia-smi` reports it, driver, P2P state (`nvidia-smi topo -p2p r`), topology (`nvidia-smi topo -m`), BAR1, PCIe link, host RAM, power cap, and GPU unlock state (on, off or unknown, as `merlin doctor` prints it; see "Optional host upgrade" above: a card unlocked since its last power-off is still unlocked with the module unloaded);
2. the preflight output and the launcher used (with any `KV_BYTES` change);
3. the boot: time to ready and the logged pool (`GPU KV cache size: ... tokens`);
4. single-stream decode at 4K, 131K and 262K, and the N-way aggregate at the profile's `max-num-seqs`;
5. the 262K needle test and a smoke chat (text, thinking, a tool call);
6. anything that failed: the first error line and the card it came from.

A one-command bench, identity smoke, needle and pool-readout script for this does not exist yet; until then the list above is the contract.

## Design constraints

The quad stays TP2xPP2+EP, 25,23, seqs 8, fp8 KV, pin 4.1e9. No quant is lowered to make a platform fit, and nothing here uses NVMe
(a host-RAM-only expert tier for 2 x 24 GB cards is not implemented).

## `merlin serve`: one front end for every checkpoint

`merlin serve [checkpoint] [checkpoint-dir] [--profile ID | --tp N] [--instances N] [--dry-run] [-- extra vllm args]`
picks a profile and a card set, then starts the server through the one launcher, `tools/merlin/launch.py`. `merlin
list [checkpoint] [--arch sm_80|sm_86|sm_89] [--cards N] [--vram GB]` prints the checkpoint -> profile -> status ->
hardware -> launcher-file table, `merlin doctor [args]` runs `tools/merlin/preflight.py`: it honours `GPUS` /
`CUDA_VISIBLE_DEVICES` like serve, otherwise reports the card set `merlin serve` would pick, refuses the same rows
(P2P-qualified rows without P2P, pins under the launcher's 1 GiB minimum) and, on a live host, applies serve's
available-RAM check.

- **Checkpoint.** A name (`flash-next`, `swift1.5`, `swift1.5-huihui`, `27b`, `27b-thinkingcap`, `nemotron-super-vl`, `ornith1.5`)
  or just the directory: `config.json` and the recipes' `checkpoint.sha256` lists identify it (27B and ThinkingCap share
  every small file, so that pair is taken from the directory name with a note; pass the name to be explicit).
- **Profile.** With no `--profile`, the largest **tested** layout that fits a homogeneous card class on this host; if no
  tested row matches the hardware (every non-3090 card today), the largest fitting untested row, labelled
  `untested / community-validated` (or 🟡 testing where its runs passed on cloud-hosted cards of that kind; a testing row
  is picked exactly like an untested one). `--tp N` restricts to that tensor-parallel size; `--profile ID` names a row (it is
  still checked against the hardware). The untested octo rows therefore stay behind `--profile fn-octo-3090-tp2pp4` on
  an 8 x 3090 host with P2P, where the default is one tested quad on GPUs 0-3 plus a hint for GPUs 4-7, and
  `--instances N` prefers the layout with the most tested instances (16 x 3090 `--instances 2` = two tested quads, not
  two untested octos). Without P2P a single launch takes the largest fitting untested row and
  8 x 3090 (or 32) gets one octo server on eight cards (`--instances N` for quads). A row with `"auto_select": false`
  is a **named alternative**: it is never the automatic pick while another row of its class fits, whatever its label,
  and runs with `--profile`. The Flash-Next 4 x 24 GB no-P2P default stays TP2xPP2 (`fn-quad-3090-nop2p`, sm_89
  `fn-quad-24g-sm89`), which balances prefill and decode best; the PP4 x TP1 rows
  `fn-quad-3090-nop2p-pp4` (tested on the reference host with P2P off: prefill +47-53 %, real-text decode -14 %)
  and `fn-quad-24g-sm89-pp4` (untested) are named alternatives.
- **Cards.** `GPUS=` / `CUDA_VISIBLE_DEVICES=` always win. Otherwise a layout smaller than the host takes a subset of
  the idle cards (a card holding more than 1 GiB is skipped; the refusal names it and the `GPUS=` that forces it):
  tested-name cards first, then the same model together, then by topology (NUMA node, host bridge, PCIe switch, from
  `nvidia-smi topo -m` and `/sys/bus/pci/devices/<bus>/numa_node`), then lowest indices; P2P-connected per
  `nvidia-smi topo -p2p r` when the row needs peer-to-peer (a partially connected host still gets a connected pair).
  Free cards of the same class are printed with
  the exact command for another instance (`GPUS=4,5,6,7 PORT=8001 merlin serve ...`). Cards of another class are
  named as skipped: a layout never spans classes. Same-class card mixes (RTX 3090 + RTX 3090 Ti, A10 + A10G, A6000 + A40)
  are one class and may share a layout; the run is labelled tested only when every card matches the row's tested
  names, it runs at the slower card's pace, and the mixed 3090 + 3090 Ti quad on the Flash-Next family carries one
  public community report (2 x 3090 + 2 x 3090 Ti, PCIe 3.0, the v2.5.1 engine: [HF discussion](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin/discussions/1)).
- **`--instances N`.** N disjoint sets of the profile's size, cut contiguously over the same order (tested names
  together, same model together, then NUMA node / host bridge / PCIe switch, then index), so on a 16-card two-socket
  board every set stays inside one node and a 3090 + 3090 Ti pool gives same-model sets where the counts allow (a set
  that still mixes models carries the mixed-card note on its own instance). The 4-card pair shape keeps the reference
  host's measured stride cut (4 x 3090, 2 x TP2 -> GPUs 0,2 and 1,3) unless the topology says the contiguous cut is
  tighter; busy cards are skipped. When the cut is not P2P-connected where the row needs it, a bounded search over the
  idle cards finds any partition that is (past the bound a P2P-needing row refuses and says to start by hand).
  Instances span every card class: each class gets its own row of the layout (the largest layout of which N fit across
  the classes, or the `--tp` / `--profile` one), tested classes first, then the larger VRAM class — on
  8 x RTX 4090 24 GB + 8 x RTX 4090 48 GB, `--tp 1 --instances 16` starts eight `27b-tp1-sm89-48g` and eight
  `27b-tp1-sm89-24g`; a layout never spans classes, a class with no row is named in a note, and when fewer than N fit
  the refusal lists what every class holds.
  Each instance gets its own consecutive `PORT` with `MASTER_PORT` / `VLLM_PORT` derived from it (`distributed_ports`
  in `launch.py`: never equal to any HTTP port of the block or to another instance's), its own cache root after the
  first (`CACHE_ROOT`, `VLLM_CACHE_ROOT` and `TRITON_CACHE_DIR` under `.vllm-cache/instance-<i>`), its own log
  (`merlin-<profile>-<port>.log`, or under `MERLIN_LOG_DIR`) and its own process group; `--served-suffix` appends `-<i>`
  to the served names (the checkpoint's own names for Swift and Huihui). Each instance's environment and argv equal
  what `GPUS=<set> PORT=<p> merlin serve` produces plus those ports and the per-instance cache paths (a single Flash-Next or 27B
  launch sets no `MASTER_PORT` / `VLLM_PORT`; its cache variables are the launcher's own: Flash-Next exports
  `VLLM_CACHE_ROOT`, plus `TRITON_CACHE_DIR` when warm, under its cache root, and 27B exports none; `CACHE_ROOT=<dir>`
  maps to both on every family); `-- --port` / `--host` are refused with `--instances`. `merlin serve` refuses a
  Flash-Next launch when the host memory available now is below the layout's resident floor (~69 GiB at four workers
  plus ~2 GiB per worker above four: a TP2xPP4 octo needs ~77 GiB); `--instances` needs that for every new instance
  and sizes the plan inside that budget (running servers are already out of the available figure; against MemTotal
  they are counted too). "Available" is MemAvailable capped by what the cgroup has left, with the reclaimable page
  cache (`inactive_file`) not counted as used. `merlin doctor` applies the same check on a live host; the recipe shims
  warn on the memory available now instead of refusing, as the hand-written launchers did, but refuse when the host RAM (MemTotal
  or the cgroup limit) is under the resident floor. Free-card hints name only a second
  server that passes preflight on its cards and fits the host RAM beside the first, with the instance's own
  `CACHE_ROOT`, `MASTER_PORT` and `VLLM_PORT`. The port checks take IPv4 and IPv6 hosts.
- **Reduced pools.** A card under the VRAM its row's pin was sized for (A10 / A10G / L4 report 22.5 GiB against the
  24 GiB class) gets a KV pin lowered by the shortfall (whole blocks where the row records them), printed as
  `reduced pool` with the estimated token count and 262K / 131K residency; quality is unchanged. Rows that keep a
  smaller class's pin (the sm_89 32 / 48 GB 27B and Super VL rows keep the 24 GB pin, `serve.pin_vram_gib`) lose
  nothing on an L40S at 45 GiB; rows sized for their own class (A40 against the sm_86 48 GB pin) do. A pin that holds
  no full 262K session, or that falls under the launcher's 1 GiB minimum, is refused.
- `--dry-run` prints the resolved cards, environment, launcher invocation and the final `vllm serve` line and starts
  nothing (`--simulate 4x3090` reuses preflight's hardware spec: `<N>x<card>[:opts]`, parts joined with `+` for a
  mixed host, up to 32 cards, e.g. `8x4090+8x4090-48g`, `2x3090+2x3090ti`, `16x3090:numa=2:switch=4:p2p=node`;
  `:numa=K` / `:switch=S` add `topo -m`-style links, `:p2p=node|switch` limits peer access to one node / switch, and
  `:p2p` / `:nop2p` / `:bar1=` / `:used=` on a part apply to its cards). Extra arguments after `--` go to `vllm serve` last on
  every checkpoint (vLLM keeps the last copy of a flag). A `--speculative-config` there (any spelling vLLM accepts: `--speculative_config`, `-sc` or `--speculative-config.KEY`) is merged into the recipe's;
  any other flag that overrides a recipe value prints a note, naming the knob (MAXLEN, SEQS, CHUNK, LPT) that sets it
  with the launcher's checks.

## The one launcher: `tools/merlin/launch.py`

Every launcher file in the repository is a shim for `tools/merlin/launch.py --profile <id>`: `models/flash-next/serve.sh`
(`fn-quad-3090`), `models/swift1.5/serve.sh`, `models/swift1.5-huihui/serve.sh`, the three 27B launchers, the two
ThinkingCap wrappers (`--checkpoint 27b-thinkingcap`), `models/nemotron-super-vl/serve.sh` and the generated
`tools/merlin/launchers/*.sh`. The launcher reads the profile from `tools/merlin/profiles/*.json` (every per-profile
value lives there), applies the environment knobs the recipes always took, and runs the shared start-up logic once:
the venv-toolkit `CUDA_HOME` default and the first-serve toolchain checks, the build-product re-hash, the PLE-dtype
check of `config.json`, the `vllm/` shadow refusal, the host-RAM / `/dev/shm` warnings, the `AUTO_TOPO` topology
selector (four-card P2P-on Flash-Next-family rows), a port-free check, and `SERVE_PRINT_ENV_AND_EXIT=1` (prints the platform
decision and the vllm flags instead of starting). The Flash-Next family runs the full set as the v3.0.0 recipe did; the
27B and Super VL families run the `CUDA_HOME` default only (their recipes never checked build products or the toolchain,
and the tested launch commands must stay identical). `CHUNK` (`--max-num-batched-tokens`), `BLOCK_SIZE` (`--block-size`) and `LPT`
(`--long-prefill-token-threshold`) are knobs on every profile; the defaults are today's values.


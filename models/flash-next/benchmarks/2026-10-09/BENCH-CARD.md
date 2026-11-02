# v3.0.0 bench card — 2026-10-09 (against v2.5.1)

> **Historical record — v3.0.0 (0.30) figure throughout.** Every measurement and comparator below belongs to the
> original vLLM 0.30 check, not the 0.32 engine. See the [Merlin 3.0.0 summary](../2026-10-15/BENCH-CARD.md) for current results.

> *File names in this record (`serve/serve.sh`, `scripts/...`, `release/...`, `patches/...`) are from the measurement time; current locations are in [models/README.md](../../../README.md).*

> **Provenance.** Reference host: 4× RTX 3090, 220 W, driver 610.43.02. The v2.5.1 profile figures and the 393K YaRN figures
> were measured on 2026-10-09; the v3.0.0 figures come from the release check run on 2026-10-10 on the same host. Source: the
> v3.0.0 engine, a pre-release engine build, 2026-10-10, served by `scripts/serve.sh` with its
> defaults, the historical release venv and the published checkpoint. Check verdict: 16 of 17
> items PASS; the VRAM-headroom item failed on boot mode (v3.0.0 compiled fresh, v2.5.1 loaded its cache) and passed on a
> warm-against-warm re-run the same day. The release check measures the shipped configuration as a whole; it does not attribute speed or quality changes to individual candidates.

## 1. Decode ladder against v2.5.1

Single stream (c=1), MTP K=3, 4×3090 TP2×PP2, 3 prompts per cell, v3.0.0 on 2 boots against the 2 boots of v2.5.1's release
ladder with the same prompts. Tokens per event = output tokens per streamed event. The whole release is measured, not each
change separately.

| prompt tokens | thinking | v3.0.0 tok/s | v2.5.1 tok/s | change | median ITL ms (v3.0.0 / v2.5.1) | tokens per event (v3.0.0 / v2.5.1) |
|---|---|---|---|---|---|---|
| 4,096 | on | 186.1 | 157.9 | +17.8 % | 16.43 / 19.44 | 3.02 / 3.03 |
| 32,768 | on | 187.0 | 159.2 | +17.5 % | 16.63 / 19.59 | 3.04 / 3.04 |
| 131,072 | on | 193.6 | 161.8 | +19.7 % | 16.72 / 19.85 | 3.12 / 3.13 |
| 261,120 | on | 206.2 | 174.3 | +18.3 % | 17.02 / 20.03 | 3.39 / 3.39 |
| 4,096 | off | 135.9 | 117.4 | +15.8 % | 16.35 / 19.36 | 2.21 / 2.27 |
| 32,768 | off | 136.1 | 115.2 | +18.1 % | 16.59 / 19.49 | 2.23 / 2.24 |
| 131,072 | off | 138.3 | 117.6 | +17.7 % | 16.79 / 19.87 | 2.30 / 2.31 |
| 261,120 | off | 134.3 | 117.6 | +14.2 % | 17.09 / 20.04 | 2.28 / 2.32 |

Pooled per-boot index (thinking-on cells): +18.31 % (95 % CI −2.53 to +39.16 %, 2 boots a side; the release bar was a
difference of at least −1.5 %). Against v2.5.1 booted the same day on the same host (1 boot): +12.83 % on the thinking-on
cells, with median ITL 16.43 / 16.63 / 16.72 / 17.02 ms against 18.95 / 18.98 / 19.15 / 19.55 ms at the four depths.

For reference, the 2026-10-09 decode profile of v2.5.1 (untraced) put a step at 19.18 / 19.63 / 19.98 ms at 4K / 131K / 258K
with 3.29 / 3.30 / 3.17 tokens per step.

Where a v2.5.1 step goes (node trace at 131K; 4K and 258K within ±0.2 ms): the two pipeline stages run serially, stage 0
computing 7.0 ms and stage 1 11.0 ms, with 2.0 ms per step where neither computes (host launch and stage hand-off), independent
of depth. On stage 1 the dense GEMMs take about 3.9 ms, the BF16 vocabulary head 2.25 ms (5 calls per step), Marlin MoE 1.3 ms,
custom all-reduce 0.6 ms, QSA attention 0.66–0.89 ms and GDN 0.25–0.35 ms. From 4K to 258K depth adds only about 0.6 ms.

### v2.5.1 reference on the reference host (single stream, thinking off, temperature 0, 512 tokens; medians of 3 prompts)

| depth | prefill tok/s | TTFT s | decode tok/s | follow-up TTFT s |
|---|---|---|---|---|
| 32K | 5,286 | 6.1 | 146.8 | 0.38 |
| 131K | 5,314 | 24.6 | 137.9 | 0.79 |
| 258K | 5,111 | 50.5 | 139.1 | 1.14 |

Concurrent cold 16K prompts, then simultaneous 512-token decode (aggregate tok/s, cold makespan): 2 requests 204 tok/s, 10.4 s;
4 requests 371 tok/s, 13.0 s; 8 requests 598 tok/s, 25.2 s. GSM8K-200 (5-shot, thinking off, temperature 0): 198/200.

### v3.0.0 on the same cells and prompt bytes (release check)

| depth | prefill tok/s | TTFT s | decode tok/s | follow-up TTFT s |
|---|---|---|---|---|
| 32K | 5,396 | 5.9 | 159.1 | 0.37 |
| 131K | 5,407 | 24.2 | 160.5 | 0.77 |
| 258K | 5,219 | 49.5 | 148.3 | 1.12 |

Concurrent cold 16K prompts, then simultaneous 512-token decode: 2 requests 266 tok/s, 6.7 s; 4 requests 442 tok/s, 12.6 s;
8 requests 659 tok/s, 25.0 s. GSM8K-200 (same instrument): 198/200, the same items as v2.5.1 (exact McNemar p = 1.0).

## 2. Resident capacity

| entry | KV pool | resident sessions | preemptions / errors |
|---|---|---|---|
| native 262K (default) | 924,993 tokens | 3 × 262K or 6 × 131K | 0 preemptions (v3.0.0: 3 × 258,022 and 6 × 126,949 prompt tokens held resident, the next request not admitted) |
| 2×393K YaRN (optional static-YaRN entry) | 952,199 tokens (2.42× one 393K session) | 2 × ~380K | 0 preemptions |

The 393K entry's check (PASS): needle found exactly at about 385K at three positions; GSM8K-20 19/20, the same as the default
entry. Static YaRN factor 1.5 applies at every sequence length on that entry, so the native 262K entry remains the default.

## 3. Image cap

The cap stays at 4 MP per image at the full 4.1e9 KV pin. Measured on the reference host: 4 MP keeps today's per-card VRAM spare; 8 and
12 MP serve correctly but drop the stage-0 cards to 464 MiB free (below the 512 MiB floor); 14 MP runs out of memory in the
vision tower. For 16.8 MP images use a smaller KV pin.

## 4. Idle power and start-up

| measure | v3 |
|---|---|
| idle power per card after defined idle interval (cards at P8) | 22.5–33.0 W; all four cards at P8 from 18 s after the last response through a 180 s tail, after a client abort and a burst with aborts |
| start-up time, launch to the API answering | 268 s with an empty compile cache; 158–172 s on later boots (3 boots). v2.5.1 from its cache, same host and day: 153 s |

`VLLM_E2_GUARD_IDLE` (default 1 in the historical engine) parked the bounds-guard reader once the engine drained so idle cards could drop to P8. On vLLM 0.32 this switch is retired; idle parking is built in (see [current running notes](../../docs/reference.md#running-notes)). The stock
v2.5.1 control sat at about 106–122 W per card idle on the reference host.

## 5. Decode candidates

> **Current-default correction.** The historical table's INT4 reduced-vocabulary head row is not a current shipping-default claim.
> The current launcher defaults to a full-vocabulary BF16 draft head (`DRAFT_HEAD=bf16`, `REDUCED_DRAFT_VOCAB=0`);
> no qualified reduced-vocabulary table ships. The old check does not establish the current reduced-head configuration.
> Sources: [`launch.py`](../../../../tools/merlin/launch.py) (`_build_flash_next`),
> [`registry.py`](../../../../tools/merlin/registry.py) (`DRAFT_HEAD_DEFAULT`) and the [current reference](../../docs/reference.md#stack).

| change | status | result |
|---|---|---|
| RecoverSSM metadata reuse (always on) | **ships** (always on) | Ships in the release configuration; the release check does not isolate its individual decode effect. |
| INT4 reduced-vocabulary draft head | **not shipping** (removed; see the correction above) | Not in the release configuration; no decode result is claimed. |
| GDN verify launch configuration | **off by default** (engine option; `scripts/serve.sh` does not enable it) | Off in the release configuration, so the release check does not measure it; no decode result is claimed. |
| Persisted Triton autotune | **ships** (on by default; `WARM=0` opts out) | start-up time in section 4 (together with the boot warmup trims); the release check does not isolate its decode effect. |
| Boot warmup trims | **ships** (on by default; `WARM=0` opts out) | start-up time in section 4 (together with the autotune warm start) |
| Local-argmax mixin (brandon-nuttall's GitHub contribution) | **opt-in** (off by default) | Opt-in and excluded from the shipped-configuration release check; no per-change quality or decode result is claimed. |
| Fused QSA draft metadata | **not shipping** | no speed gain: +0.3 to +1.0 % median ITL over 3 boots, inconclusive |
| Fine-grained prefix cache | **not shipping** | residency held (6 × 131K + 3 × 262K, 0 preemptions); multi-turn hit rate 0.862, same as without it |
| Larger image cap (8–16.8 MP at the full pin) | **not shipping** | see section 3 |

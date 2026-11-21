# Qwen3.8-27B — community bench card, 2026-11-15: 2 x RTX 3080 20 GB (modded), dynamic-BAR1 P2P

> **Provenance.** Contributed by [brandon-nuttall](https://github.com/brandon-nuttall); community measurements, not
> the maintainers'. Host: 2 x modded RTX 3080 20 GB
> (GA102, PCI ID `10de:2206`, vBIOS 94.02.27.00.14), same PCIe root complex, no NVLink, PCIe Gen4.
> Driver 615.71.09 open kernel modules with the dynamic-BAR1 P2P patch (`noon-at-cgn/geforce-dynbar1-p2p`):
> byte-verified P2P integrity PASS (264 cases / 2,112 transfers / 1,403.6 GiB, SM store/load both ways),
> peer copy 26.4 GB/s one-way / 51.2 GB/s bidirectional. Engine: Merlin 3.0.2.1 (vLLM 0.33), served with
> `merlin serve`, profile `27b-tp2-sm86-20g`, TP2, P2P=on (NCCL P2P transport), 48 GiB RAM KV tier.
> Checkpoint: [halt95/Qwen3.8-27B-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-27B-W4A16-Merlin),
> pinned revision `7a0af39f63dd4fb0733c23d3ec126762074a503f`. Streams are client-timed; raw logs retained by the contributor.
>
> Two driver configurations were benchmarked on identical hardware — the older static-BAR1 595.91.07
> kit with P2P=off (NCCL host route), and the dynamic-BAR1 615.71.09 stack with P2P=on. All numbers below
> are the same scripts on both stacks, so the deltas are attributable to the P2P route (and its driver).

## 1. Single-stream decode (4,096-token thinking-on output, client-timed)

| prompt | 615 dynbar1, P2P=on | 595 static, P2P=off | delta |
|---|---|---|---|
| ~19 K (countdown) | 231.1 / 233.5 / 234.4 tok/s | 103.9 / 105.8 / 107.8 tok/s | ~2.2x |

## 2. Decode ladder (512-token output, median of 3 sends, cold cache)

| context | 615 dynbar1, P2P=on | 595 static, P2P=off | delta |
|---|---|---|---|
| 4,136 | 134.9 tok/s | 115.4 tok/s | +17% |
| 131,056 | 101.6 tok/s | 92.2 tok/s | +10% |
| 261,316 | 80.1 tok/s | 50.6 tok/s | +58% |

Rung-1 numbers are cold; warm runs sit higher on both stacks (the 4 K row spans 90-143 on the new stack, consistent with the reference-host variance notes).

## 3. Cold salted prefill scaling (unique salt per rung, max_tokens=1)

| prompt tokens | 615 dynbar1, P2P=on | 595 static, P2P=off | delta |
|---|---|---|---|
| 39,094 | 1,585 tok/s | 885 tok/s | +79% |
| 78,406 | 1,361 tok/s | 810 tok/s | +68% |
| 117,694 | 1,188 tok/s | 750 tok/s | +58% |
| 157,008 | 1,055 tok/s | 697 tok/s | +51% |
| 235,608 | 864 tok/s | 610 tok/s | +42% |

## 4. 8-agent overlap (8 x ~16 K conversations, 8 simultaneous 512-token continuations)

| stack | aggregate decode |
|---|---|
| 615 dynbar1, P2P=on | 425.6 tok/s (4,096 tok / 9.6 s) |
| 595 static, P2P=off | 240.1 / 289.8 tok/s (two runs) |

## 5. GSM8K-200 (thinking off)

Identical scores: 191/200 (1 cut at token limit, 0 errors) on both stacks; wall time 264 s (P2P=on) vs 331 s (P2P=off).

## 6. Power (NVML, per-card mean under load, pair sum)

| phase | 615 dynbar1, P2P=on | 595 static, P2P=off |
|---|---|---|
| idle | ~46 W | ~61 W |
| prefill (~190 K) | ~609 W | ~523 W |
| decode (4,096-out) | ~607 W | ~560 W |

Decode energy per token halves on the P2P stack (~2.6 vs ~5.3 J/token at the 19 K rung).

## Driver notes for sm_86 20 GB pairs

- The dynamic-BAR1 615.71.09 driver (`noon-at-cgn/geforce-dynbar1-p2p`) is a patched third-party stack that Merlin
  does not ship or test; on the static-BAR1 595-era stack keep P2P off, as reported below.
- The 595-era static-BAR1 route shows copy-engine-only P2P (bandwidth check passes, SM peer dereference
  returns all-ones); NCCL P2P transport then hangs the first allreduce during warmup. The dynamic-BAR1
  615 route is byte-verified end to end.
- With the 615 stack the boot-time `gpu_unlock` BF16 probe is a no-op (phase-2 returns -EBUSY), and BF16
  GEMM measures at full rate: fp16 57.4 / bf16 64.5 TFLOPS (GPU0), fp16 61.8 / bf16 64.7 (GPU1) — no
  GeForce BF16 handicap observed.
- One warm-reset hazard: a single warm BMC reset onto the loaded 615 stack wedged the host's NIC and
  journaling while leaving the console alive; power actions were accepted but deferred. Cold cycles were
  reliable throughout. Repro conditions filed with the fork.

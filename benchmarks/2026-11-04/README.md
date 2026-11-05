# Merlin 3.0.2 bench cards — 2026-11-04

One card per checkpoint that was re-run for 3.0.2, measured on the reference host on vLLM 0.33.0. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock.
Each card has the single-stream decode ladder at 4K and 131K with thinking on and off (the Swift1.5 card at 4K, 32K, 131K
and 259K), the exact 262K row
(261,888 prompt tokens + 128 out) and cold prefill, compared with the 3.0.1 card. The Flash-Next and Qwen3.8-27B (2 and 4
GPUs) cards also carry a fresh 8-agent row with thinking on; the Nemotron 3.5 Super VL, Ornith 1.5 and Swift1.5 8-agent runs were
not repeated for 3.0.2 (their 3.0.1 cards have them). The Flash-Next card also has a same-session quality A/B against 3.0.1. Swift1.5-Huihui and ThinkingCap were not re-run; their latest cards are the
[3.0.1 cards](../2026-11-02/README.md).

| card | checkpoint | GPUs | compared with |
|---|---|---|---|
| [Qwen3.8-Flash-Next](Flash-Next-BENCH-CARD.md) | Qwen3.8-Flash-Next | 4 | 3.0.1 (2026-11-02) |
| [Swift1.5](Swift1.5-BENCH-CARD.md) | Swift1.5 | 4 | 3.0.1 (2026-11-02; run 2026-11-05) |
| [Qwen3.8-27B](Qwen3.8-27B-BENCH-CARD.md) | Qwen3.8-27B | 1, 2, 4 | 3.0.1 (2026-11-02) |
| [Nemotron 3.5 Super VL](Super-VL-BENCH-CARD.md) | Nemotron 3.5 Super VL | 4 | 3.0.1 (2026-11-02) |
| [Ornith 1.5](Ornith-1.5-BENCH-CARD.md) | Ornith 1.5 | 4 | 3.0.1 (2026-11-02) |

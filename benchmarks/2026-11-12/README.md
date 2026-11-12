# Merlin 3.0.2.1 bench cards — measured 2026-11-09 to 2026-11-12

One card per checkpoint, measured on the reference host on vLLM 0.33.0, mostly on 2026-11-10 and 2026-11-11 (UTC): Flash-Next, Swift1.5 and Swift1.5-Huihui on the published 3.0.2.1 wheel (each card names the rows that were not), the other four on the previous build of that wheel (built on 2026-11-10, before the prefill runs change), except ThinkingCap's two-card exact-262K rows, re-measured on the published wheel. Each card dates its rows and names their build; sessions held at once, the 8 MP image answers and the 131K / 262K needle panels are from the 3.0.0 cards. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock.
Each card has the single-stream decode ladder with thinking on and off (the Flash-Next card at 4K and 131K, the others also at 32K and
their deepest depth), the exact 262K row (261,888 prompt tokens + 128 out) and cold prefill, compared with the 3.0.2 card, plus
8-agent rows with thinking on and off and GSM8K-200 where the layout runs them. The Flash-Next card also has the 4 x 20 GB row (emulated on 3090s) and the
quality panel (the full GSM8K test set and the needle panel).

| card | checkpoint | GPUs | compared with |
|---|---|---|---|
| [Qwen3.8-Flash-Next](Flash-Next-BENCH-CARD.md) | Qwen3.8-Flash-Next | 4 | 3.0.2 (2026-11-04) |
| [Swift1.5](Swift1.5-BENCH-CARD.md) | Swift1.5 | 4 | 3.0.2 (2026-11-04; run 2026-11-05) |
| [Swift1.5-Huihui](Swift1.5-Huihui-BENCH-CARD.md) | Swift1.5-Huihui | 4 | 3.0.2 (2026-11-04; run 2026-11-05) |
| [Qwen3.8-27B](Qwen3.8-27B-BENCH-CARD.md) | Qwen3.8-27B | 1, 2, 4 | 3.0.2 (2026-11-04) |
| [ThinkingCap](ThinkingCap-BENCH-CARD.md) | ThinkingCap | 2, 4 | 3.0.2 (2026-11-04; run 2026-11-05) |
| [Nemotron 3.5 Super VL](Super-VL-BENCH-CARD.md) | Nemotron 3.5 Super VL | 4 | 3.0.2 (2026-11-04) |
| [Ornith 1.5](Ornith-1.5-BENCH-CARD.md) | Ornith 1.5 | 4 | 3.0.2 (2026-11-04) |

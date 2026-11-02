# Merlin 3.0.1 bench cards — 2026-11-02

One card per checkpoint, measured on the reference host. Measured on 4 x RTX 3090 at 220 W with P2P enabled (aikitoria's open-gpu-kernel-modules P2P driver) and Zagitta's gpu-unlock.
Each card has the single-stream decode ladder from 4K to 261K with thinking on and off, the exact 262K row
(261,888 prompt tokens + 128 out), cold prefill and agents at once, compared with the 3.0.0 card, and GSM8K-200 at reasoning effort medium.

| card | checkpoint | GPUs | compared with |
|---|---|---|---|
| [Qwen3.8-Flash-Next](Flash-Next-BENCH-CARD.md) | Qwen3.8-Flash-Next | 4 | 3.0.0 (2026-10-20) |
| [Swift1.5](Swift1.5-BENCH-CARD.md) | Swift1.5 | 4 | 3.0.0 (2026-10-20) |
| [Swift1.5-Huihui](Swift1.5-Huihui-BENCH-CARD.md) | Swift1.5-Huihui | 4 | 3.0.0 (2026-10-20) |
| [Qwen3.8-27B](Qwen3.8-27B-BENCH-CARD.md) | Qwen3.8-27B | 1, 2, 4 | 3.0.0 (2026-10-20) |
| [ThinkingCap](ThinkingCap-BENCH-CARD.md) | ThinkingCap | 2, 4 | 3.0.0 (2026-10-20) |
| [Nemotron 3.5 Super VL](Super-VL-BENCH-CARD.md) | Nemotron 3.5 Super VL | 4 | 3.0.0 (2026-10-20) |
| [Ornith 1.5](Ornith-1.5-BENCH-CARD.md) | Ornith 1.5 | 4 | 3.0.0 (2026-10-20; run 2026-10-21) |

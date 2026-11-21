# Merlin 3.0.2.2 bench cards — measured 2026-11-21

Every lane was spot-checked on 3.0.2.2 on the reference host, on vLLM 0.33.0: boot, smoke checks, one 4K thinking-on
decode row and one 100K prefill row against the 3.0.2.1 card and that row's run-to-run band, the KV pool and the needle
panel. The lanes inside every band keep their [3.0.2.1 cards](../2026-11-12/README.md), carried forward. The four lanes
below were outside a band, and their decode ladder, exact 262K row and cold prefill were run again. 8 agents and GSM8K-200
in both thinking modes were run again on the 27B and ThinkingCap four-card rows and on Ornith 1.5 (the 27B four-card
thinking-on GSM8K-200 is carried); the Flash-Next 4 x 20 GB row has neither, as on 3.0.2.1 (its pool holds six of the
eight agents). The spot-check figures of the
carried lanes are in the [changelog](../../docs/CHANGELOG.md#measured-on-merlin-3022), with the Hy4 Flash rows; the
others are in each card.

| card | checkpoint | GPUs | compared with |
|---|---|---|---|
| [Qwen3.8-Flash-Next](Flash-Next-BENCH-CARD.md) | Qwen3.8-Flash-Next | 4 x 20 GB (emulated) | 3.0.2.1 (2026-11-12) |
| [Qwen3.8-27B](Qwen3.8-27B-BENCH-CARD.md) | Qwen3.8-27B | 4 | 3.0.2.1 (2026-11-12) |
| [ThinkingCap](ThinkingCap-BENCH-CARD.md) | ThinkingCap | 4 | 3.0.2.1 (2026-11-12) |
| [Ornith 1.5](Ornith-1.5-BENCH-CARD.md) | Ornith 1.5 | 4 | 3.0.2.1 (2026-11-12) |

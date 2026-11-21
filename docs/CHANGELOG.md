# Changelog

Every Merlin release in one place, newest first. Until 3.0.0 each model shipped from its own repository with its own
version numbers; here each sits under the engine release it ran on, with its original tag named. Figures under the 2.x
releases were measured on vLLM 0.28-0.30 when they shipped and are kept as published; they are not re-measured on 3.0.0.
Figures under 3.0.0 and 3.0.1 were measured on vLLM 0.32; the 3.0.2 figures are measured on 0.33 except those marked as carried over from 3.0.1. The 3.0.2.1 figures are measured on 0.33 on every row of its table; each bench card lists the earlier panels it carries over. 3.0.2.2 re-measured four lanes on 0.33 and carries the rest from 3.0.2.1 after a spot check (‡).

| Release | Date | Engine | Models |
|---|---|---|---|
| [3.0.2.2](#merlin-3022) | 2026-11-21 | vLLM 0.33.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5, Hy4 Flash |
| [3.0.2.1](#merlin-3021) | 2026-11-12 | vLLM 0.33.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.2](#merlin-302) | 2026-11-05 | vLLM 0.33.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.1](#merlin-301) | 2026-11-02 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [3.0.0](#merlin-300) | 2026-10-22 | vLLM 0.32.0 | Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap, Nemotron 3.5 Super VL, Ornith 1.5 |
| [2.5.3](#v253) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next, Swift1.5-Huihui (new); Qwen3.8-27B v1.3.0 / v1.3.1 and ThinkingCap (new) |
| [2.5.2](#v252) | 2026-10-12 | Merlin engine v2.5.2 (vLLM 0.30.0) | Flash-Next |
| [2.5.1](#v251) | 2026-10-06 | vLLM 0.30.0 + Merlin commits | Flash-Next, Swift1.5 |
| [2.2.0](#v220) | 2026-09-25 | vLLM 0.30.0 + 58 commits | Flash-Next, Swift1.5 (new); Qwen3.8-27B v1.0.0 (new) |
| [2.0.1](#v201) | 2026-09-18 | vLLM pre-0.30 | Flash-Next |
| [2](#v2) | 2026-09-17 | vLLM pre-0.30 | Flash-Next |
| [1](#v1) | 2026-09 | vLLM 0.28.0 + patches | Flash-Next |

## Merlin 3.0.2.2

2026-11-21. Built on vLLM 0.33.0, like 3.0.2.1. The seven models of 3.0.2.1 keep their recipes (the VERSION files are
unchanged), and Hy4 Flash is new, as a `tools/merlin/launch.py` lane on four RTX 3090s (testing: full-model runs passed, kernel parity
qualification pending) and eight (untested on hardware). The compiled CUDA code is byte-identical to 3.0.2.1's and the Rust frontend is rebuilt only for the new
version string; the Python side of the engine, the launchers and setup change. This release fixes a hang in the Qwen3
parser at the end of an output, changes the whitespace a reply's content starts with and keeps the whitespace beside
tool calls, opens a Qwen3 tool call only at a line start and never inside code, makes forced and strict tool calls on
the Qwen tool parser reliable and bounds what a strict call can cost the server, changes four KV RAM cache defaults,
turns on the 27B four-card P2P decode settings, returns KV-capacity failures as client errors, adds start-up checks
that refuse an install they cannot verify, and adds Hy4 Flash with its decode speed-ups and a choice of FP8 or BF16 Engram packs.

### Highlights

- **The Qwen3 parser no longer loops at the end of an output when tools are offered.** This is the parser behind the
  `qwen3` reasoning parser and the `qwen3_coder` tool-call parser, which Flash-Next, Swift1.5, Swift1.5-Huihui,
  Qwen3.8-27B, ThinkingCap and Ornith 1.5 serve together, in streaming and non-streaming replies. (Nemotron 3.5 Super VL
  uses the same tool-call parser with its own reasoning parser; the hang tests cover the `qwen3` pairing.)
  - **The cause:** the parser holds back a possible `<tool_call>` opener until it can tell whether it is real, in up to
    three layers (an open code fence, a quote in the reasoning, and the tool-name check). If the output ended while an
    opener was still held, the end-of-output step let the layers hand it back to each other and never finished.
    Parsing runs on the API event loop, so the server then stopped answering every other request too.
  - **When it happened:** with tools in the request and the output ending with an opener still held: reasoning that
    ends in `<tool_call>` (with or without a partial `<function=` header); an unclosed code fence in the reasoning
    followed by a tool call; content inside an open code fence that ends in `<tool_call>` (with or without a partial
    header).
  - **Since when:** 3.0.2.1, where the parser keeps quoted and fenced tool calls as text (`MERLIN_QWEN3_LITERAL_SPANS`,
    on by default). With `MERLIN_QWEN3_LITERAL_SPANS=0`, 3.0.2.1 finished every case these tests record.
  - **The fix:** at the end of an output each hold layer is flushed at most once; a layer that takes items after its
    flush raises instead of looping, and the parser is left with no half-flushed state. A code fence that closes and another that opens inside the same piece of text is recognised as
    closed, so for these inputs reasoning and tool calls no longer depend on how a stream was split into pieces. A bare
    `<function=` at the end of a fenced block stays text.
- **Content starts at its first visible character after the reasoning block, in streamed and non-streamed replies
  alike.** A model that reasons writes `...</think>` followed by whitespace and then the answer; that whitespace is now
  dropped, with or without tool calls. For example, content that was `"\n\nAnswer"` in 3.0.2.1, streamed or not, is now
  `"Answer"`. When the reasoning block is followed directly by the answer, nothing changes. When the prompt already
  closed the reasoning block (thinking off), or the parser has no reasoning, nothing is removed. Whitespace later in the
  content is not touched. **This changes output bytes in both modes:** a client that relied on the leading whitespace
  (to separate the answer, or to compare replies byte for byte) must adjust.
- **Whitespace beside tool calls is kept, and text after a tool call is returned.**
  - **Non-streaming replies:** the content around tool calls keeps the model's own whitespace; it is dropped only when
    all of it is whitespace and the reply has tool calls. Text the model writes after a tool call is now returned
    (before, a non-streaming reply dropped it). **These change non-streaming output bytes.**
  - **Streaming replies:** the whitespace between the text and a tool call was already kept in a stream, and still
    is; the non-streamed content now matches it (for example `"Answer\n\n"` before the call, where 3.0.2.1's non-streamed
    reply gave `"Answer"`). Streamed content beside a tool call still changes when whitespace follows the reasoning
    block, through the content-start change above (`"\n\nAnswer\n\n"` becomes `"Answer\n\n"`). A stream holds a run of
    whitespace until the next text or the end of the output decides it, and the last reasoning piece takes back the
    whitespace held before it.
  - **Tool arguments whose schema also allows a string** stay strings unless the text is exactly the JSON text of a
    number, boolean or null; a value the model wrote as a JSON string stays a string, also when a streaming converter
    fails. The same holds on the `gemma4` tool-call format.
- **A Qwen3 tool call opens only at a line start, and never inside a code fence or an inline code span.** This is the
  `qwen3` parser, with `MERLIN_QWEN3_LITERAL_SPANS` on (the default; `0` leaves all of it as in 3.0.2.1).
  - **The position rule:** a call opener (the `<tool_call>` token or a bare `<function=` header) starts a call only at
    the start of the answer, at the start of a line after optional whitespace, or right after another call. Anywhere
    else (a blockquote, a list item, the middle of a sentence) it is text, and so is everything after it; a header right
    after such an opener is the rest of the quote. The decision is made when the opener arrives, so a stream holds
    nothing more for it. Reasoning keeps its own rules. A `</think>` kept in the content (no reasoning parser) still
    starts the answer, so a call written right after it opens.
  - **Code:** inside an open code fence an opener is now always text, also when the output ends inside the fence. This
    replaces the fence hold and its end-of-output rule, under which a block that ended an unclosed fence could still
    become a call. An opener inside an inline code span (a run of backticks closed by the next run of exactly as many on
    the same line) is held with what follows it until the span closes, and the whole span is then text. If the line, the
    reasoning, the turn or the output ends first, the held items take the path they took before, so a real call stays a
    call. A reply that explained the format and wrote `<tool_call>` in a markdown table cell used to lose the rest of
    the reply and return a later example as a phantom call; it now returns the table verbatim and the one real call.
  - **Constrained calls and other parsers:** a request whose calls a structural tag constrains (a named
    `tool_choice`, `"required"` or strict tools) keeps the opener token as the call wherever the grammar placed it. Only
    the Qwen3 parser applies the position rule; the parsers built on it keep their call positions.
  - **Special tokens:** every added token the tokenizer flags special (other than the think and tool-call tokens) is
    ignored when testing for a line start, so `skip_special_tokens` no longer decides whether a header after such a
    token on a line opens a call. A special token the engine drops (one the config does not name, such as a vision or
    pad token) no longer holds back the stream; before, it held every later delta to the end of the stream.
  - **Checked:** re-parsing 600 recorded replies with quoted tool markup (Flash-Next and the 27B) returned no call from a
    blockquote, list, mid-line, inline code or fence, kept every turn-start and line-start call, and kept the quoted
    text; the recorded replies kept all 168 calls.
- **Forced, required and strict tool calls on the Qwen tool parser open and close with the special tokens.** The tool
  grammar named the `<tool_call>` opener as text, so the model could spell it in ordinary tokens; the parser then read
  the call as text and the reply came back with no tool call (seen on Flash-Next). The grammar now uses the token ids
  from the model's vocabulary for `tool_choice="required"`, a named `tool_choice` and tools with `"strict": true`. When
  the four delimiter tokens cannot be found unambiguously in a tokenizer, the text form is kept with one warning. Three
  differences in free text follow from this:
  - an opener token followed by prose instead of a call is refused by the grammar;
  - an opener spelled in ordinary characters is plain text, not a call;
  - a think tag spelled in ordinary characters is allowed, as text.
- **Strict string parameters with `maxLength` are bounded by a character budget.** On a strict Qwen tool, a top-level
  string parameter with `maxLength` is now written as one grammar rule that counts characters (a character spelled in
  byte tokens counts as one), cannot contain the call's closing sequence and accepts only well-formed UTF-8. Before, the
  strict grammar unrolled the bound into one state per character. Parameters and schemas the rule cannot take keep the
  generic strict grammar, and requests without `"strict": true` build the grammar they always built. Strict stays
  opt-in. Measured on 3.0.2.2 as shipped, with the length table below, on Flash-Next (4 × RTX 3090, 2026-11-19):
  - **Decode, strict against default, ms per output token:** 6.55 against 6.39 at one stream (1.03×), 14.57 against
    14.57 at 8 streams (1.00×) and 27.66 against 25.77 at 24 streams (1.07×). The strict mode of 3.0.2.1 took 32.23 ms
    per token at one stream (5.00× its default of 6.45; 3.0.2.1-era figure, 2026-11-15).
  - **First request with a new bounded-string schema:** about 0.22 s longer to its first token than a repeat of the
    same request (new schema 0.479 s at the median, repeat 0.254 s), against about 0.7 s longer in 3.0.2.1.
  - **The first tool request after boot** still pays about 2.7 s more than a repeat (2.975 s against 0.249 s).
  - All 14 strict decode sends were valid, and every one of 71 strict sends stayed within its `maxLength`.
- **Tool-call grammars: string bounds from a length table, one whitespace run per gap, and a slow-fill guard.** A mask
  is filled on the engine-core thread, one request after another, so one tool-call request with slow fills holds every
  stream. Four changes bound that cost; none changes what a call may say.
  - **A length table for string bounds (`VLLM_MERLIN_STRICT_LENGTH_TABLE`, on by default).** xgrammar cannot cache the
    token masks of the character-budget rule above, so near the bound one fill cost up to a few hundred milliseconds. The
    backend now compiles the tag without the budget and applies the bound itself: a per-tokenizer table of how many code
    points each token adds to a value, a per-request tracker that mirrors the tag and advances and rolls back with the
    matcher, and a mask that is the grammar's mask AND the table's for the characters left. It covers top-level
    strings, strings inside JSON parameters (nested objects, arrays, `minLength` with `maxLength`), whole argument
    objects and plain strings; shapes it does not model keep their bound in the grammar and are counted. The masks equal
    the budgeted grammar's at every state (the recorded calls, bounds 0 to 200 on the five lane tokenizers, and random walks
    with drafts and rollback). `VLLM_MERLIN_STRICT_LENGTH_TABLE=0` compiles the tag as given.
  - **One whitespace run per gap, and a cap on it (`VLLM_MERLIN_STRICT_WS_CAP`, 64).** xgrammar writes each
    one-property JSON part with a whitespace run on each side, so two unbounded runs met between parameters and a run of n
    characters had n + 1 parses; the matcher kept them all and its memory grew with the square of the run (4.5 GB at
    15,360 characters on a workstation, an out-of-memory kill on a CPU test host with the cap off). The grammar is now
    compiled with one run per gap, with and without the length table, and 15,360 characters stay under 50 MiB.
    The cap bounds each run at 64 characters at a site and 128 where one parameter ends and the next site starts;
    string values are not capped. Non-strict tags and `response_format` grammars do not change.
  - **One parse for the padding around a value.** Inside a parameter tag xgrammar pads a value with a whitespace run on
    each side, and a value that can be empty (a string pattern such as `^[a-z/._]*$`) split a whitespace-only text between
    the two runs in L + R + 1 ways: a fill took 300 ms at about 90 characters. The grammar now writes these values with
    one parse per text, with the same accepted text, for: an anchored sequence of simple atoms none of which can match
    whitespace; a class pattern `^[C]*$` whose class holds whitespace and not `<`; and a string-or-null union (exactly
    `anyOf` string and null), which is written as the string because xgrammar pads a union but not a plain string. A tool
    with no bounded string keeps its arguments as one JSON-schema element, and only its padded value fragments are
    replaced. A tool without an enforced schema, called with `tool_choice="required"` or a named tool, gets xgrammar's
    generic argument object; in its four XML-style argument formats each value now has one
    parse for the whitespace at its start and end (before, a run of 12,288 characters took 2.9 GB; the tests hold it
    under 50 MiB). Anything the rewrite cannot read back compiles as given.
  - **A slow-fill guard (`VLLM_MERLIN_STRICT_STALL_GUARD`, on by default), a backstop.** Each mask fill of a tool-call
    request that takes at least 300 ms of CPU time on the calling thread (`VLLM_MERLIN_STRICT_STALL_FLOOR_MS`) counts
    toward that request's slow-fill time over its last 64 fills (`VLLM_MERLIN_STRICT_STALL_WINDOW`). At 5000 ms
    (`VLLM_MERLIN_STRICT_STALL_BUDGET_MS`) the request ends with an error, as a request whose tokens the grammar rejects
    does (the client gets `Internal server error`), and other requests continue; a log line and a counter record each
    end. Healthy fills stay under the floor and add nothing. The recorded whitespace stall ends at fill 131 of 138; the
    earlier rule of four slow fills in a row never fired on it, and that silence is kept as a regression test.
    `VLLM_MERLIN_GRAMMAR_STEP_DIAG=1` (off by default) writes one NDJSON line per engine step with the accept, trim and
    per-row mask fill times and the GPU time between the forward pass and sampling.
  - **Still slow, and ended by the guard:** two value shapes (see Known limits).
- **Qwen3.8-27B and ThinkingCap on four RTX 3090s with P2P (`27b-tp4-3090`) turn on four decode settings:**
  `VLLM_DRAFT_BOUNDED_PLAN`, `VLLM_GDN_DECODE_FASTPATH`, `VLLM_GDN_SHARED_ALIGN_INDEX` and `VLLM_TARGET_SEQ_LENS_ASYNC`
  default to 1, as on the two-card row; the two-card all-reduce settings are not set on this row. Your own value of each
  is kept. Measured on 4 × RTX 3090 with P2P (four-arm A/B/B/A): one-stream milliseconds per output token 13.0 to
  17.9 % lower on five of six cells (4K, 131K and 262K, thinking on and off; the sixth, 262K thinking on, 11.6 % inside
  a wider run-to-run band), eight streams unchanged within the band; temperature-0 output identical on 28 of 28 prompts,
  the KV pool (2,012,105 tokens) and the image floor unchanged. On ThinkingCap at 4K: 16.6 % (thinking on) and 16.8 %
  (thinking off) lower, output identical. The other four-card rows and the one-card row set none of them.
- **DFlash2 takes its own prefill-runs tick costs** (`SPEC=dflash2`; the Flash-Next profile's `base.spec_tick_costs`):
  the launcher exports `VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS=289.0,22.6,0.4` (prefill tick ms, decode round ms, fill
  ticks) for DFlash2 only, instead of the engine default `139,25,1` that was measured on the MTP row, and only at the
  chunk they were measured at, 2048 (the DFlash2 rows' default): the first number is the cost of one full-chunk step, so
  with another `CHUNK` the launcher exports nothing, the engine default applies and one start-up note says so. Measured
  on 4 × RTX 3090 (chunk 2048, 24/24 split): the sessions decode 0.296 of the time during a long ingest (target 0.30,
  band ± 0.030) against 0.162 with the default; temperature-0 output was identical on 28 of 28 prompts, the KV pools were
  unchanged and the image floor held. **The cost: a cold prompt's first token comes about 18 to 19 % later** (64K 11.7 to
  13.9 s, 128K 23.8 to 28.2 s). The longest pause between streamed tokens during a 64K ingest stays about 0.82 s on
  DFlash2 with or without the values (0.82 s before, 0.83 s after; MTP: 0.41 s). Every MTP row, and `SPEC=none`, keeps
  the engine default and launches with the same arguments and environment as before; an exported
  `VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS` still wins at every chunk and for both drafters, and the launcher's note says
  which source the engine got. DFlash2 stays opt-in. The variable is a runtime-only knob, so no compile or AOT cache is
  invalidated by the upgrade.
- **A request the KV cache cannot hold gets HTTP 400 with the cause, not 500.** Both scheduler capacity paths (a lone
  running request that cannot grow, and a request that can never fit) now answer as a client error carrying the logged
  cause. A streaming `/v1/responses` request that fails after `response.created` ends with a Responses `error` event,
  and a failed background (stored) response carries its `error`.
- **KV RAM cache: duplicates-first and waiting-last eviction, 8 assembling units and 64 copy credits are now the
  defaults.** These act only in engines booted with the host tier (`--kv-ram-cache`, `--swarm`); an explicit value of
  each variable still wins.
  - **Duplicates first** (`MERLIN_KV_HOST_TIER__EVICT_DUPLICATES_FIRST`, was 0): when the host cache needs room, a unit
    whose exclusive rows all duplicate GPU blocks in the free pool is given up after the running-twin duplicates and
    before the host-only units. The guard is checked on the rows each take actually frees, at the take and again at
    apply, and keeps a unit off the early class while a GPU twin lies within the walk's reclaim lookahead. An admission
    that would take the GPU twins of a dropped host copy is held for at most two walks while the unit is copied first,
    then goes ahead. A copy given up this way is copied again if the GPU later needs that space before the session
    returns.
  - **Waiting last** (`MERLIN_KV_HOST_TIER__EVICT_WAITING_LAST`, was 0): a request in the waiting queue claims the plain
    hashes of its prompt at arrival and gives them up at admission or when it finishes or is aborted while waiting. Host
    eviction takes every unclaimed unit first, in the existing order, then the claimed units, the furthest back in the
    queue first. A unit is claimed by its exclusive rows only; leases, restoring rows and running twins keep their
    protection. Nothing walks the queue per step.
  - **Eight assembling units** (`MERLIN_KV_HOST_TIER__MAX_ASSEMBLING`, was 2) and **64 copy credits**
    (`MERLIN_KV_HOST_TIER__COPY_CREDITS_D`, was 32). The cold-start hold cap stays a third of the GPU pool at any slot
    count (a ninth at 8 slots would have refused the first preempt grace of a run on a 3 x 262K pool); the largest
    recorded unit still replaces it once one exists.
  - **The restore switches:** `MERLIN_KV_HOST_TIER__RESTORE_CREDITS_PER_SLOT` is on (measured alone on the 27B at 96 GB
    with the KV RAM cache on, it cost no hit rate: 98.90 % of the trace's ceiling, no unguarded losses).
    `MERLIN_KV_HOST_TIER__RESTORE_START_ANY_STEP` and `MERLIN_KV_HOST_TIER__RESTORE_SEAT_HOLD` are off, and each is
    opt-in with `1`: the start switch alone costs about 1.9 percentage points of hit rate and adds unguarded losses, and
    the hold needs it. `MERLIN_KV_HOST_TIER__SHARED_ROW_LONE` stays off (experimental): when set, a block with several
    owners is copied on its own only when the union of its owners would never fit the host cache; any union that fits
    fans out to the owners' members as with the switch off.
  - **Diagnostics:** `MERLIN_KV_TIER_DIAG=1` adds lines (see Engine).
- **Hy4 Flash ships with FP8 or BF16 Engram tables.** The FP8 Engram pack stays the default; the BF16 Engram pack
  keeps the n-gram tables at their source precision, at 118 GiB of pinned host RAM instead of 60 GiB (the same
  660,006,416 rows of 96 values at 192 B per row instead of 98 B). `./setup.sh --model hy4flash --engram bf16`
  sizes the host RAM check (about 198 GiB available before the start, against 140 GiB for FP8), the memlock check
  and the download for it and starts the launcher with `ENGRAM=bf16`; on `tools/merlin/launch.py` the `ENGRAM`
  knob names the pack and is checked against the checkpoint's `config.json` (`engram_fp8_embedding` true = FP8,
  false or absent = BF16). With `ENGRAM` unset the FP8 launch is unchanged. The BF16 Engram pack is untested on hardware:
  it needs about 198 GiB of host RAM available, more than our four-card host has.

### Measured on Merlin 3.0.2.2

4× RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W; runs of 2026-11-20 and 2026-11-21 (UTC), on trees that carry this release's engine code; the only later non-comment engine change is the MTP layer-index guard on pipeline-parallel layouts, and Ornith 1.5, the one lane it affects, was re-measured on it.
Every lane was spot-checked on 3.0.2.2 (boot, smoke checks, one 4K thinking-on decode row and one 100K prefill row against its 3.0.2.1 figure and that row's run-to-run band, the KV pool and the needle panel). A lane inside every band keeps its 3.0.2.1 figures, marked ‡ below; a lane outside a band had the affected sections (decode ladder, exact 262K, agents) run again, with cold prefill, and the agents and GSM8K-200 rows in both thinking modes where the layout runs them, alongside.
- decode is one stream, thinking on (effort medium where the template has levels), median of 3 (ThinkingCap 2× at 262K: median of 12);
- 8 agents is the total of eight simultaneous follow-ups on 16K conversations.

The 20 GB row is measured on 4 × RTX 3090 that emulate 20 GB cards (4 GiB held back per card), not on real 20 GB cards.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next ‡ | 4× 24 GB | 924,993 | 174 / 188 / 192 | 6,839 tok/s | 640 / 641 † | 197 / 198 |
| Flash-Next | 4× 20 GB (emulated) | 265,888 | 153 / 159 / 171 | 4,838 tok/s | n/a (pool held 6 of 8 on this run) | n/a |
| Qwen3.8-27B ‡ | 2× | 826,689 | 248 / 197 / 160 | 1,117 tok/s | 740 / 677 | - / 194 |
| Qwen3.8-27B | 4× | 2,012,105 | 315 / 296 / 267 | 2,118 tok/s | 1,012 / 857 | 198 ‡ / 195 |
| Swift1.5 ‡ | 4× | 924,993 | 180 / 185 / 182 | 6,793 tok/s | 688 / 661 | 199 / 198 |
| Swift1.5-Huihui ‡ | 4× | 924,993 | 178 / 190 / 203 | 6,893 tok/s | 719 / 668 † | 198 / 197 |
| Qwen3.8-27B ‡ | 1× | 271,506 | 131 / 102 / 75 | 587 tok/s | n/a (two at most) | - / 194 |
| ThinkingCap ‡ | 2× | 826,689 | 244 / 179 / 163 | 1,116 tok/s | 731 / 663 | - / 195 |
| ThinkingCap | 4× | 2,012,105 | 287 / 277 / 274 | 2,118 tok/s | 997 / 875 | 197 / 194 |
| Nemotron 3.5 Super VL ‡ | 4× | 1,856,367 | 105 / 102 / 101 | 4,193 tok/s | 444 / 417 | 196 / 194 |
| Ornith 1.5 | 4× | 894,618 | 152 / 106 / 106 | 2,302 tok/s | 522 / 481 | 195 / 192 |

‡ 3.0.2.1 figures, carried forward: the lane (or, in a 4× cell, that one figure) was not re-measured on 3.0.2.2. The 20 GB row has no GSM8K run; the one- and two-card rows have no thinking-on GSM8K-200 run (`-`), as in 3.0.2.1.
† The Flash-Next and Swift1.5-Huihui thinking-off 8-agent cells are from 3.0.2.1's previous build: their runs on the published 3.0.2.1 wheel ended early (one reply stopped by itself; overlap windows of 0.25 and 0.29 s) and were not used.

The spot checks of the carried lanes (4K thinking-on decode and 100K prefill on 3.0.2.2, against 3.0.2.1 and that row's band):
- **Flash-Next 4× 24 GB:** 171.6 against 174.0 tok/s (band ±3 %); 6,999 against 7,036 tok/s (band ±3 %).
- **Swift1.5:** 178.4 against 179.8 tok/s (band ±3 %); 6,999 against 6,964 tok/s (band ±3 %).
- **Swift1.5-Huihui:** 178.9 against 178.4 tok/s (band ±3 %); 7,046 against 7,076 tok/s (band ±3 %).
- **Qwen3.8-27B 2×:** 248.7 against 248.4 tok/s (band ±3 %); 1,670 against 1,687 tok/s (band ±3 %).
- **Qwen3.8-27B 1×:** 130.6 against 130.7 tok/s (band ±3 %); 893 against 891 tok/s (band ±3 %).
- **ThinkingCap 2×:** 243.7 against 244.0 tok/s (band ±3 %); 1,674 against 1,679 tok/s (band ±3 %).
- **Nemotron 3.5 Super VL:** 98.3 against 104.8 tok/s (−6 %, inside this row's band of ±9 %); 5,111 against 5,082 tok/s (±3 %).

Every carried lane booted to the same KV pool as on 3.0.2.1.

Against 3.0.2.1:
- **Qwen3.8-27B 4× and ThinkingCap 4×:** decode +20 % at 4K thinking on (315.5 against 262.0 and 287.2 against 240.1
  tok/s), +15 % and +17 % at 131K, and +15 % and +18 % at exact 262K (267.2 and 274.2 tok/s; median of 3 sends,
  266.7–277.6 and 265.2–276.0); thinking off +10 % to +20 % at every depth. The gain is the time per decode step, from
  the four-card P2P decode settings now on by default: a median event interval of 12.8 ms against 15.4 ms at 4K, with
  the tokens per event level. Over 6 sends at exact 262K the 27B gives 275.4 tok/s thinking on (266.7–299.7) and
  181.1 tok/s off (173.0–205.7). Cold prefill 0 % to +1 %; eight agents thinking on +4 % and +2 % (short overlap windows,
  3.41 and 3.61 s), thinking off 0 % and +2 % (857 and 875 tok/s). GSM8K-200 thinking off 195/200 and 194/200
  (3.0.2.1: 195/200 and 193/200).
- **Flash-Next 4× 20 GB:** decode +4 % / 0 % at 4K / 131K thinking on and 170.8 tok/s at exact 262K (+3 %; sends
  163.4–176.8); cold prefill +5 % to +6 % (the 3.0.2.1 20 GB rows were measured on that release's previous build,
  before its prefill runs change).
- **Ornith 1.5:** thinking on, decode is level with 3.0.2.1 at 4K (152.1 against 148.7 tok/s) and 105.8 tok/s at
  exact 262K (−1 %; sends 101.7–112.2). At 131K the median is 106.4 tok/s (−13 %; sends 97.9–124.5):
  two of the three replies stopped by themselves at about 187 tokens, with fewer tokens per step, and six other
  131K texts on the same engine path gave 117.5 tok/s (115.9–121.0; −4 % against 3.0.2.1's 122); with MTP the number of tokens per
  step follows the text generated. Thinking off −5 % to +4 %. Cold prefill +1 %; eight agents +1 % thinking on and −1 %
  off. GSM8K-200 195/200 thinking on and 192/200 off (3.0.2.1: 195/200 and 191/200).
- **ThinkingCap 4×:** GSM8K-200 thinking on 197/200 (3.0.2.1: 196/200).

**Hy4 Flash (new):** measured on 2026-11-20 and 2026-11-21 on the same install, with its own driver (not the frozen
ladder above): one stream, thinking on at effort high (the lane's served default), T=0, each send a cold prefill,
256 tokens out (128 at exact 262K), median of 3 sends; decode tok/s = (completion tokens − 1) / (last − first token
time). DFly2 drafts; K=3 is the shipped default.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 128K / 262K | 262K prefill | 8 agents | GSM8K-200 |
|---|---|---|---|---|---|---|
| Hy4 Flash (`hy4flash-quad-3090`) | 4× | 609,398 | 75.7 / 67.1 § / 89.3 § | 1,384 tok/s | not run | not run |
| Hy4 Flash (`hy4flash-octo-3090`) | 8× | 4,196,751 (estimate) | untested on hardware | untested | untested | untested |

§ Measured at K=5 (the boot of the K sweep that ran every row), not at the shipped K=3. At 4K: K=3 75.7 tok/s (13.19 ms per token: median event
interval / tokens per step), K=5 75.3 (13.33 ms), K=7 68.1 (14.77 ms), DFly2 off 45.1 (22.05 ms). 128K is a
130,816-token prompt (sends 66.3–71.4 tok/s), 262K is exactly 261,888 tokens (sends 77.6–89.7). Cold prefill
1,672 tok/s at 10,240 tokens and 1,608 tok/s at 102,400; 262K time to first token 189.3 s. The 8-card
pool is the block arithmetic of the 4-card row, not a boot.

Every row except Hy4 Flash is in the bench cards; the Hy4 Flash rows are in this section and the `hy4flash-quad-3090` profile evidence. Measured on 3.0.2.2: [Flash-Next 4× 20 GB](../models/flash-next/benchmarks/2026-11-21/BENCH-CARD.md) · [Qwen3.8-27B 4×](../models/qwen38-27b/benchmarks/2026-11-21/BENCH-CARD.md) · [ThinkingCap 4×](../models/thinkingcap/benchmarks/2026-11-21/BENCH-CARD.md) · [Ornith 1.5](../models/ornith1.5/benchmarks/2026-11-21/BENCH-CARD.md). Carried from 3.0.2.1: [Flash-Next](../models/flash-next/benchmarks/2026-11-12/BENCH-CARD.md) · [Swift1.5](../models/swift1.5/benchmarks/2026-11-12/BENCH-CARD.md) · [Swift1.5-Huihui](../models/swift1.5-huihui/benchmarks/2026-11-12/BENCH-CARD.md) · [Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-12/BENCH-CARD.md) · [ThinkingCap](../models/thinkingcap/benchmarks/2026-11-12/BENCH-CARD.md) · [Nemotron 3.5 Super VL](../models/nemotron-super-vl/benchmarks/2026-11-12/BENCH-CARD.md).

### Before you upgrade

- **New wheel.** The wheel is `vllm-0.33.0+merlin3.0.2.2.c56eeed392-cp313-cp313-linux_x86_64.whl`; `release/merlin-wheel.sha256` names it, so this checkout refuses the 3.0.2.1
  wheel. The wheel's file name now carries the release (`...+merlin3.0.2.2...`), so it differs from the 3.0.2.1
  wheel's. Its package description is now a fixed text, and it declares `jinja2 >= 3.1.6` (the version the release
  lock already pins). Compiled files embed their build date, so a rebuild of the same sources is not byte-identical:
  check a wheel by its published sha256 (`release/merlin-wheel.sha256`), not by rebuilding it.
- **Stop the server before you update.** The engine is installed editable, so `git pull` changes the source a running
  server imports. Every server started from the checkout now holds `.merlin/serve.lock`, and `./setup.sh` refuses to
  reinstall while one runs. A server started by the 3.0.2.1 launcher does not hold the lock: restart it once after
  updating.
- **The launchers check the installed native code at every start, for every model family.** A tree without
  `.build-source` (an install that `./setup.sh` or `release/install-env.sh` never recorded) is refused until
  `./setup.sh` records it, and stale or modified native sources or changed build products are refused with the files to
  commit, remove or repair. A bare checkout used only to launch an installed wheel, and a copy without Git metadata, are
  judged on their recorded product hashes and start. `MERLIN_SKIP_NATIVE_CHECK=1` skips the native-source and FlashInfer
  checks, with a warning on every start; see [release/README.md](../release/README.md).
- **Three switches accept only `0` or `1`.** `MERLIN_REPACK_RELEASE`, `MERLIN_LOAD_PEAK_LOG` and
  `MERLIN_QWEN3_LITERAL_SPANS` stop the start with an error naming the variable for any other value (`true`, `yes`, an
  empty value). Unset keeps the default.
- **Only the literal loopback addresses count as local.** `127.0.0.1` and `::1` serve without an API key and without a
  warning; `localhost` is a name and no longer counts. Setup saves a `localhost` answer as `127.0.0.1` and says so, and generates a key for any other
  bind. `merlin serve` with `HOST=localhost` and no key prints the no-authentication warning. A saved specific address
  (for example an IPv6 address or `::`) is kept when the Access question is asked again, offered as `keep`.
- **The 27B and Nemotron 3.5 Super VL launchers refuse `--disable-chunked-mm-input` with a long-prefill cap** (`LPT` /
  `--long-prefill-token-threshold` above 0): an image longer than the cap could stall scheduling. Remove the flag or set
  `LPT=0`.
- **Output bytes change** (see Highlights): content after the reasoning block no longer starts with whitespace, in
  streamed and non-streamed replies; non-streamed content keeps the whitespace before a tool call; non-streamed replies
  return text written after a tool call. Clients that compare replies byte for byte with 3.0.2.1, or that relied on the
  leading whitespace, will see these differences; tool calls themselves are unchanged.
- **`/v1/messages` and `/v1/responses` streams now get the SSE keep-alive** that `merlin serve` turns on (a `: keep-alive`
  comment line every 15 s while a stream sends nothing), as `/v1/chat/completions` and `/v1/completions` already did.
  `-- --sse-keep-alive-interval 0` turns it off.
- **The DFlash2 drafter's folder follows `--models-dir`** (Flash-Next family, `--spec dflash2` only). With
  `--models-dir DIR`, a new download goes to `flash-next-dflash2` beside `DIR` instead of `~/models/flash-next-dflash2`.
  A drafter already in `~/models/flash-next-dflash2` keeps being used while there is none beside `DIR`, so nothing is
  downloaded again; when both exist the one beside `DIR` is used. The start-up note names the folder it takes.
  `DFLASH2_MODEL` still overrides both. With the default `~/models/<model>` both places are the same folder.
- **Metrics:** `vllm:merlin_kv_host_tier_census_events_total{event="restore_failed"}` (and the write-back log's
  `restore_failed` field) now counts restore failures with the census off, where it stayed 0 before; the gauge series
  `vllm:merlin_kv_host_tier_census{observation="host_rows_duplicate_running"}` and
  `{observation="host_rows_duplicate_imminent"}` are removed (never measured, always 0); a failed store copy now logs
  `SimpleCPUOffload: store copy for event N failed; the host copy is dropped` instead of the restore wording.

### Fixes

- **Streamed content after the reasoning no longer depends on where the stream is cut.** In 3.0.2.1, when whitespace after the reasoning
  block was followed by text and a tool call, the streamed content changed with the delta boundaries: cut inside or
  right after that whitespace, the whitespace moved to the end of the text (`"\nAnswer\n\n\n"` or `"Answer\n\n\n\n"` instead of
  `"\n\nAnswer\n\n"`). The content is now the same at every cut.
- **A stuck KV RAM cache lock now fails with a message after 900 s instead of hanging the boot.** The tier's ranks
  take a host-wide lock (`/tmp/merlin-kv-host-tier.lock`) one at a time while they allocate their arenas. A process that
  held it for good, or a second account on the same host, used to block every later boot with no output, or die with a
  bare `PermissionError` when it tried to create the file. The file is now opened read-only when it exists (any account
  can lock it), created only when it is missing, refused when it is a symbolic link or not a regular file, and waited
  for at most `MERLIN_KV_HOST_TIER__ARENA_LOCK_TIMEOUT_S` seconds (default 900; the measured wait of a four-rank 96 GiB
  boot is 64 s). A rank that has waited 5 s logs one line saying so, and each rank logs how long it held the lock. On a
  timeout the error names the file, its owner and how to find the holder (`fuser` / `lsof`). Nothing changes for a boot
  that is not blocked.
- **KV RAM cache admission:** a copy the write-back refuses no longer frees a host row first; a request whose total
  need is zero is not charged the full admission reservation; copy-on-write refusals and the restore retry bounds are
  kept on host-tier admission, and a returning request with no work left is logged once and skipped. None of these
  changes output tokens.
- **`/v1/responses` refuses two different tools that would reach the model under one name** (for example a function
  and a namespace member that flatten to the same `<namespace>__<name>`) with a 400 that names it. The same tool listed
  twice is not refused.
- **`/v1/messages` merges adjacent turns of the same role** before converting them, so a history split into several
  user or assistant turns gives the same prompt as the one-turn form. Block order and `tool_use` ids are kept, and
  system turns stay boundaries.
- **`/v1/completions` counts cached prompt tokens once per prompt:** repeated stream chunks and the `n > 1` choices of
  one prompt no longer add to the cached count twice.
- **`prompt_logprobs` with the top-20 bound** is admitted with a smaller shared chunk where it was refused at the full
  chunk; a request that cannot fit on its own is still refused.
- **A non-FP8 KV cache uses unit k/v scales** on compressed-tensors checkpoints (a bfloat16 or float16 cache, or a layer
  skipped with `--kv-cache-dtype-skip-layers`), instead of the checkpoint's FP8 scales. The shipped rows serve FP8 KV
  and are unchanged.
- **Server load tracking releases each request once.** A request whose client disconnected could be released twice,
  which lowered the server-load figure below the real load.
- **`merlin serve --instances N` probes readiness over HTTPS** when the server is started with TLS, and refuses a
  readiness probe it cannot make before it starts any server.
- **Setup keeps a specific bind address** instead of widening it, writes IPv6 endpoints in brackets, and a failed
  install restores the checkout's build products and native records together with the previous venv (they are copied
  to `.merlin/products-old-*` for the length of the install).
- **`merlin doctor`** builds every report section from one configuration, refuses a report path that is not a regular
  file, and its GPU probe has a deadline. Before installing, setup checks again, under the install lock, that the
  checkout is still the planned commit with no modified tracked file.
- **The native-build check warns** when the FlashInfer the server will import is not the recorded patched copy or cannot
  be verified, and resolves a bare Rust-frontend name on the `PATH` the engine gets. Both only warn.
- **The 27B four-card profiles advertise 131K session counts that fit the pool:** 14 instead of 15 on `tp4-sm80-24g`
  and 62 instead of 63 on `tp4-sm80-80g` (estimates).
- **The manual downloads in the model READMEs pin the revision setup downloads.** The Flash-Next, Qwen3.8-27B and
  ThinkingCap READMEs and the engine README ran `hf download` without `--revision`, so their `sha256sum -c` step failed
  once the Hugging Face repository moved on. Every manual download in the docs now names the revision that the model's
  `checkpoint.sha256` and `release/download-sizes.json` record, the same one `./setup.sh` fetches.
- **The DFlash2 drafter follows `--models-dir`.** `./setup.sh --spec dflash2` put the drafter in
  `~/models/flash-next-dflash2` whatever `--models-dir` said. Setup and `merlin serve --spec dflash2` now use
  `flash-next-dflash2` beside the checkpoint folder; `DFLASH2_MODEL` still wins, and a drafter already in
  `~/models/flash-next-dflash2` is still found (see Before you upgrade). With the default models folder nothing moves.
- **Speculative decoding with sampling: the resample insert kernel launches with one warp.** The kernel that inserts a
  resampled token (`_insert_resampled_kernel`) reads the count of sampled tokens and stores that count plus one in
  place. Launched with Triton's default four warps, a delayed warp can read the incremented count and index past the
  request's logits rows, which faults on the temperature load; upstream reports `Warp Out of range Address` core dumps
  with Gemma 4 MTP at four speculative tokens. The launch now passes `num_warps=1`, the change of vllm-project/vllm#66048
  (by shustovTE, in vLLM 0.34.0); the other launches in the file already did. The sampling algorithm and every other
  launch are unchanged. Sampled tokens are unchanged.
- **`P2P=ipc` needs peer access inside each tensor-parallel group only.** In ipc mode NCCL runs with
  `NCCL_P2P_DISABLE=1`, so pipeline send and receive and the expert-parallel collectives go through host shared memory;
  only the PCIe-IPC all-reduce maps peer buffers, and the engine builds it on the tensor-parallel group alone. The
  check now asks for CUDA peer access and a BAR1 of at least 1 GiB between the cards of each tensor-parallel group of the
  chosen row's layout (contiguous runs of `tp` cards in rank order), not between every selected pair.
  - `merlin serve` takes `P2P=ipc` into profile selection, so a refused request fails there; when no usual card set
    passes, an ipc row may take peer-connected groups out of order (for example `CUDA_VISIBLE_DEVICES=0,2,1,3`).
  - Refusals name the failed pair or card (`TP pair (GPU 2, GPU 3) has no CUDA peer access`,
    `GPU 5 BAR1 256 MiB < 1 GiB`); `P2P=on` and a partial matrix in auto name the pairs without peer access. The
    unreadable-matrix hint names the serve control (`P2P=ipc` on Flash-Next, `P2P=on` on the 27B and Super VL).
  - `P2P=on` keeps the all-pairs and BAR1 requirements; `on`, `off` and unset select as before.
- **Setup's 20 GB messages:** the recommendation on a host without a Flash-Next layout names the 4 x 20 GB layout, the
  cold-expert note says what the profile check found instead of a bare status word, and the untested notice is cut
  between words and never ends on a number without its unit.

### Engine

- **Parser self-test at every boot:** the served chat template and the tool and reasoning parsers are round-tripped
  once in a background helper on a private tokenizer, comparing the exact stream and the whole text. A mismatch or
  missing coverage is logged as a warning (`Parser/template self-test (...)`) and never blocks the server.
  `merlin doctor --parser-selftest <checkpoint>` runs the same check without a server. It has no off switch. Its cost,
  measured on four boots on 4 × RTX 3090 (Flash-Next with 24 agents, 27B on four cards and on two, and the Flash-Next
  20 GB-card rows emulated) and on a CPU host: the check runs on a background thread for 0.87 to 0.98 s on the four
  boots (about 0.85 s on the CPU host) and the server becomes ready without waiting for it; before start-up completes it takes a
  synchronous snapshot of 0.25 to 0.37 s (CPU host); on the CPU host the server process gains no memory, and a short-lived helper
  process peaks at about 240 MiB for about one second.
- **`/v1/models` adds `kv_cache_capacity_tokens`:** total KV cache tokens shared by all requests on this server (the
  engine's KV pool, the same figure the boot line `GPU KV cache size: N tokens` prints, e.g. 2,012,105 on 27B TP4 and
  924,993 on Flash-Next fn-quad-3090; the minimum over the engines a server routes to; null for render-only servers or
  engines without a KV pool). A single request is limited by `max_model_len`, not by this figure. It does not shrink
  with occupancy. Like the boot line, it is sized in `max_model_len`-length requests, so on hybrid models it depends on
  `max_model_len`.
- **New metric:** `vllm:merlin_scheduler_policy_events_total{policy,event}` carries prefill-run and decode-first counter
  deltas.
- **MTP draft layers** are built at their runtime index on single-stage layouts when every by-name lookup agrees
  (otherwise, and on every pipeline-parallel layout, the previous construction runs, with one warning), the draft step reaches the MTP module, and the speculative config reads one
  module count. Outputs are bit-identical on the 27B and ThinkingCap.
- **Diagnostics, all off by default and none changing a token or a policy decision:**
  - `MERLIN_SPEC_RESIDUAL_DIAG=1` counts the rejection sampler's residual draws per rank on the device (rejection
    events, residual draws with no finite mass, draws outside the support, token 0 at a rejection) and keeps a ring of
    the first 4,096 zero-mass events. A counter line goes to the engine log every 512 verify steps and at shutdown, and
    `MERLIN_SPEC_RESIDUAL_DIAG_OUT` names a directory for a per-rank JSON dump. Every rank that builds a rejection
    sampler logs one line that reads back the stable residual setting, the diagnostic switch and the sampling methods.
  - `MERLIN_KV_TIER_DIAG=1` logs each host restore as one line per event with the scheduler step and wall time
    (lease-ready, slot-start, batch-issue, batch-commit, retire, first-compute; the slot-start and first-compute lines
    carry the step the request headed the seat queue, `-1` when it never did), a restore-step line per active step (cap waiters, slots in use, issued and unissued restore rows), one line per
    `never_fit` verdict with its owners, rows and the met block, and one `dup_first_unguarded` line per unit given up early
    and then lost. That last line names the key whose loss ended the unit, whether that key was one of the early-freed
    ones (`1`, `0`, or `-1` when the loss site names no key), how many early-freed keys still have a GPU block, the
    request, the path that took the GPU twin and the stack that recorded the loss. The plans line also reports the
    waiting-last evictions by claim and queue-rank bucket, and the restore seatless-start and seat-hold counters when
    those switches are on.
  - `VLLM_MERLIN_GRAMMAR_STEP_DIAG=1` writes per-step grammar timings (see Highlights).
- **Hy4 Flash decode speed-ups, on by default on the hy4flash rows:** the hy4flash 4-card row
  (`hy4flash-quad-3090`, TP2 x PP2) turns on `MERLIN_HY4_INDEXER_FAST`, `MERLIN_HY4_SWA_WINDOW_REUSE`,
  `MERLIN_LATENT_IPC`, `MERLIN_HY4_DFLY_GRAPHS`,
  `MERLIN_HY4_INDEXER_SELECT` and `MERLIN_COLD_EXPERTS_ADAPTIVE` (that last one only with the cold-expert offload on),
  and DFly2 drafts 3 tokens instead of 5. The 4-card row also offloads cold experts by default on layers 26, 28, 30
  and 33 at hot fraction 0.25, from a calibrated profile shipped as
  `tools/merlin/profiles/expert_hot/hy4flash-p1-calibrated.json` (`MERLIN_COLD_EXPERTS=0` turns it off). The 8-card row (`hy4flash-octo-3090`) turns on the same set except
  `MERLIN_LATENT_IPC` (measured at TP2 only) and `MERLIN_COLD_EXPERTS_ADAPTIVE` (the row has no cold-expert offload).
  The TP4 DIAG row (`hy4flash-quad-3090-tp4`, new, `launch.py --profile` only) turns on none of them and offloads
  cold experts on layers 37-41 at hot fraction 0.25 from a uniform placeholder profile. `MERLIN_HY4_INDEXER_FAST`,
  `MERLIN_HY4_SWA_WINDOW_REUSE`, `MERLIN_LATENT_IPC`, `MERLIN_HY4_DFLY_GRAPHS` and `MERLIN_HY4_INDEXER_SELECT` are
  bit-identical: serial in-engine compares at temperature 0 match the stock path token for token.
  `MERLIN_HY4_SPARSE_DECODE` is opt-in: its ieee and split P modes are under accuracy qualification.
  `MERLIN_COLD_EXPERTS_ADAPTIVE` moves expert placement only: serial outputs are unchanged, and
  concurrent outputs vary run to run, as with any batch. `MERLIN_HY4_FUSED_ROUTER` is opt-in: as accurate as stock,
  not bit-identical. Export `NAME=0` to turn one off, or `SPEC_K=5` (on `launch.py`) for the earlier draft length.
- **Hy4 Flash KV pins are round byte counts with an engine fit:** the 4-card row pins 1,600,000,000 B per card
  (was 1.75 GiB) and chunks prefill at 1,024 tokens (was 2,048): 6,351 blocks of 251,904 B, an engine fit of 2 x 262K
  or 4 x 128K, 609,398 tokens measured (prefix caching on, FP8 `fp8_ds_mla`). The pin keeps at least 524 MiB free on
  every card at idle and at the warm-up peak with DFly2 at K=3 and K=7. The 8-card row (`hy4flash-octo-3090`) is configured for `max_model_len` 1,048,576 at
  10,800,000,000 B per card, an estimate of 4 x 1M (4,196,751 tokens). It is untested on hardware.
- **Bench tools you can run (no engine change):** `benchmarks/suite/run.py` runs the benches behind the published
  bench cards (single-stream decode, cold prefill, time to first token, the exact 262K row and the needle panel)
  against any OpenAI-compatible Merlin server, with the cards' prompts, output caps, validity rules and medians
  ([README](../benchmarks/suite/README.md)). `benchmarks/agentic/agentic_bench.py` runs 1, 2, 4 and 8 coding or
  research agents at once and reports output and total tok/s, prefix-cache hit share, task success and step latency,
  offline and from bundled data ([README](../benchmarks/agentic/README.md)). Both use the standard library only.
- **Tests and developer tooling (no product change):** the wheel audit requires the full member set, a valid `RECORD`
  and a scan of every member, and audits the built wheel's metadata (version, description, `Requires-Dist`,
  `Requires-Python`, no shipped test fixtures); the package description is inlined in `pyproject.toml`, so a README-only
  commit no longer changes the wheel. The knob-invariance check compares refusals and rejects incomplete checks; import
  formatting is fixed in three entrypoint files. The load-transaction
  return state in `simple_kv_offload` was renamed with no behaviour change.

### New flags and environment variables

| Name | Where | Default | What it does |
|---|---|---|---|
| `MERLIN_KV_HOST_TIER__ARENA_LOCK_TIMEOUT_S` | engine (KV RAM cache) | 900 | Seconds a rank waits for the host-wide arena lock before it stops with an error naming the file. Positive integer. |
| `MERLIN_KV_HOST_TIER__EVICT_DUPLICATES_FIRST` | engine (KV RAM cache) | **1** (was 0) | When the host cache needs room it first gives up copies the GPU still holds in its free pool, then sessions only the host holds. `0` keeps the earlier order. |
| `MERLIN_KV_HOST_TIER__EVICT_WAITING_LAST` | engine (KV RAM cache) | **1** (was 0) | When the host cache needs room it keeps the saved sessions that requests already waiting in the queue will use until last. `0` keeps the earlier order. |
| `MERLIN_KV_HOST_TIER__GUARD_REQUIRED_KEYS` | engine (KV RAM cache) | **1** (new) | With duplicates first on, a session whose host copy was given up early is also guarded on the other pages it needs that only the GPU still holds: an admission that would take them waits (at most two walks) while the session is copied again, as for the pages given up. `0` guards only the pages given up. Acts only with `MERLIN_KV_HOST_TIER__EVICT_DUPLICATES_FIRST` on. Exactly `0` or `1`. |
| `MERLIN_KV_HOST_TIER__MAX_ASSEMBLING` | engine (KV RAM cache) | **8** (was 2) | Units copied to the host at once whose landed rows are held until the unit is complete. `2` is the earlier value. |
| `MERLIN_KV_HOST_TIER__COPY_CREDITS_D` | engine (KV RAM cache) | **64** (was 32, the value of `COPY_CREDITS`) | Copy (GPU to host) rows in flight per rank. Unset, it follows `MERLIN_KV_HOST_TIER__COPY_CREDITS` when that is set explicitly and is 64 when neither is set. `32` is the earlier value. |
| `MERLIN_KV_HOST_TIER__RESTORE_START_ANY_STEP` | engine (KV RAM cache) | 0 (opt-in) | `1`: a waiting restore's copy starts on any step a restore slot is free, also while every running seat is taken, so the restored request takes the next free seat ahead of requests sent after it. Unset or `0` keeps the earlier order. Opt-in: it gains restore waits (returns reach their first token sooner) but, measured alone on the 27B at 96 GB with the KV RAM cache on, it costs about 1.9 percentage points of hit rate and adds unguarded losses. |
| `MERLIN_KV_HOST_TIER__RESTORE_CREDITS_PER_SLOT` | engine (KV RAM cache) | **1** | The restore copy budget is `COPY_CREDITS` rows per restore slot (64 by default) instead of 32 rows shared by every restore in flight. `0` shares one budget. Measured alone on the 27B at 96 GB with the KV RAM cache on, it cost no hit rate (98.90 % of the trace's ceiling, no unguarded losses). |
| `MERLIN_KV_HOST_TIER__SHARED_ROW_LONE` | engine (KV RAM cache) | 0 (experimental) | `1`: a block with more than one owner (for example the first page of a shared system prompt) is copied to the host on its own, only when the union of its owners would never fit the host cache; any union that fits fans out to the owners' members as with `0`. The owners still hold the row on landing, but a unit whose held rows are all such shared rows takes no assembling slot. Exactly `0` or `1`; not measured on GPUs. |
| `MERLIN_KV_TIER_DIAG` | engine (KV RAM cache) | off | `1` logs host-tier decisions (new in 3.0.2.2: the restore lifecycle, `never_fit` verdicts and `dup_first_unguarded` lines; see Engine). Exactly `0` or `1`. |
| `MERLIN_KV_HOST_TIER__RESTORE_SEAT_HOLD` | engine (KV RAM cache) | 0 (opt-in) | `1`: when a running seat frees while the oldest waiting request's restore copy is still in flight, the seat waits for it (one hold at a time, at most 3 s) instead of going to a request sent later, so restores keep their place in the queue. Acts only with `MERLIN_KV_HOST_TIER__RESTORE_START_ANY_STEP` on, so it carries that switch's cost. Unset or `0` keeps the earlier order. |
| `MERLIN_SKIP_NATIVE_CHECK=1` | every launcher | off | Skips the native-source and FlashInfer provenance checks, with a warning on every start. Only the exact value `1`; build-product checksums are still checked. |
| `merlin doctor --parser-selftest <checkpoint>` | `merlin doctor` | off | Round-trips a fixed reasoning and tool-call turn through the checkpoint's chat template and parsers (tokenizer files only). Takes `--reasoning-parser`, `--tool-call-parser`, `--chat-template`, `--chat-template-kwargs`. Warn-only. |
| `--sse-keep-alive-interval` | vLLM; 15 from `merlin serve` | 0 in the engine | Now also covers `/v1/messages`, `/v1/responses` and `GET /v1/responses/{id}?stream=true`. |
| `VLLM_DRAFT_BOUNDED_PLAN`, `VLLM_GDN_DECODE_FASTPATH`, `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_TARGET_SEQ_LENS_ASYNC` | 27B launcher, `27b-tp4-3090` with `P2P=on` | **1** (was unset) | Four-card P2P decode settings (see Highlights). Your own value is kept. |
| `VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS` | Flash-Next launcher, `SPEC=dflash2` at chunk 2048 | **289.0,22.6,0.4** (was the engine default) | DFlash2's prefill-runs tick costs (see Highlights). An exported value wins. |
| `VLLM_MERLIN_STRICT_LENGTH_TABLE` | engine (tools with `"strict": true`) | **1** (new) | Applies a strict string's `maxLength` from a per-tokenizer length table instead of a grammar budget (see Highlights). `0` compiles the tag as given. Only `0` and `1`. |
| `VLLM_MERLIN_STRICT_STALL_GUARD` | engine (tool-call requests) | **1** (new) | Ends a request whose mask fills keep taking long (see Highlights) with an error. `0` turns the guard off. Only `0` and `1`. |
| `VLLM_MERLIN_STRICT_STALL_FLOOR_MS` | engine (tool-call requests) | 300 | CPU milliseconds a single mask fill must take before it counts toward the guard. A whole number from 1 to 600,000. |
| `VLLM_MERLIN_STRICT_STALL_BUDGET_MS` | engine (tool-call requests) | 5000 | Counted slow-fill milliseconds at which the guard ends the request. A whole number from 50 to 3,600,000. |
| `VLLM_MERLIN_STRICT_STALL_WINDOW` | engine (tool-call requests) | 64 | Number of a request's latest fills the guard counts over. A whole number from 1 to 100,000. |
| `VLLM_MERLIN_GRAMMAR_STEP_DIAG` | engine | off | `1` writes one NDJSON line per engine step with the accept, trim and per-row mask fill times and the GPU time between the forward pass and sampling. |
| `VLLM_MERLIN_STRICT_WS_CAP` | engine (tools with `"strict": true`) | 64 | Caps each whitespace run a strict tool call may write, between parameters and inside JSON values, at this many characters (twice that after a JSON parameter, where two runs meet). A whole number from 1 to 128: the cap bounds the parser memory and mask time a long run costs, so it cannot be turned off, and 0 or a value out of range stops the start. String values are not capped. |
| `MERLIN_REPACK_RELEASE`, `MERLIN_LOAD_PEAK_LOG`, `MERLIN_QWEN3_LITERAL_SPANS` | engine | unchanged | Now only `0` or `1`; any other value stops the start. |
| `kv_cache_capacity_tokens` | `/v1/models` | n/a | The engine's KV pool in tokens (see Engine). |
| `error` (stored response), `error` event | `/v1/responses` | n/a | A failed stored response carries its error; a failed stream ends with an `error` event. |
| `vllm:merlin_scheduler_policy_events_total` | metrics | n/a | Prefill-run and decode-first counter deltas. |
| `VLLM_SPEC_STABLE_RESIDUAL` | engine | **1** (was 0) | The underflow-safe rejection-sampler residual is on by default; `0` restores the 3.0.2.1 residual. Only `0` and `1` are accepted. |
| `MERLIN_SPEC_RESIDUAL_DIAG`, `MERLIN_SPEC_RESIDUAL_DIAG_OUT` | engine | off | Debug counters for the rejection sampler's residual draws, with an optional per-rank JSON directory. Sampled tokens are unchanged. |
| `MERLIN_HY4_INDEXER_FAST` | engine (Hy4 Flash) | **1** on `hy4flash-quad-3090` and `hy4flash-octo-3090`, 0 elsewhere | `1`: the compressed indexer's FWHT and FP8 steps run as one kernel (bit-identical) and IndexPool scoring skips tiles no query can reach (same scores). |
| `MERLIN_HY4_INDEXER_FAST2` | engine (Hy4 Flash) | 0 (experimental) | `1`: experimental gated pooling in the compressed indexer, with a different pooling reduction order. Not qualified for serving; leave it off. |
| `MERLIN_HY4_INDEXER_SELECT` | engine (Hy4 Flash) | **1** on `hy4flash-quad-3090` and `hy4flash-octo-3090`, 0 elsewhere | `1`: the IndexPool group top-k's masking, packing and merge steps run as fused kernels around the same top-k and sort. Bit-identical. |
| `MERLIN_HY4_SWA_WINDOW_REUSE` | engine (Hy4 Flash) | **1** on `hy4flash-quad-3090` and `hy4flash-octo-3090`, 0 elsewhere | `1`: the sliding-window causal rows are built once per forward and reused by the later sliding-window layers with the same window. Bit-identical. |
| `MERLIN_HY4_SPARSE_DECODE`, `MERLIN_HY4_SPARSE_DECODE_HEAD_BLOCK` | engine (Hy4 Flash) | 0 (opt-in); head block 32 when the layer has 32 or more heads | `1`: sparse MLA decode and speculative verify run in Merlin's kernel (split P by default: the softmax weights in a high and a low BF16 part; or FP32 "ieee" P). Not bit-identical to stock; the ieee and split modes are under accuracy qualification, so it stays opt-in. `HEAD_BLOCK` (16 or 32) sets heads per CTA. |
| `MERLIN_HY4_FUSED_ROUTER` | engine (Hy4 Flash) | 0 (opt-in) | `1`: for batches of up to 32 rows the router's gate projection runs in a Triton kernel (FP32 accumulation, at least as accurate as cuBLAS against an FP64 reference), then the stock top-k; larger batches use the stock path. As accurate as stock, not bit-identical (a different summation order); it chose the same experts as the stock router on every row of our tests. |
| `MERLIN_HY4_DFLY_GRAPHS` | engine (Hy4 Flash) | **1** on `hy4flash-quad-3090` and `hy4flash-octo-3090`, 0 elsewhere | `1`: the DFly2 drafter's steps are captured as full CUDA graphs instead of running eagerly. Bit-identical. |
| `MERLIN_LATENT_IPC` | engine (Hy4 Flash) | **1** on `hy4flash-quad-3090`, 0 elsewhere | `1`: the LatentMoE latent all-reduce (width 1536) also runs on the PCIe IPC all-reduce instead of falling back to NCCL. Bit-identical at TP2. Not on the 8-card row (TP4, not measured). |
| `MERLIN_HY4_FUSE_IHC` | engine (Hy4 Flash) | 0 (off) | `1`: fused iHC boundaries in the Hy4 Flash decoder layer. Bit-identical to the stock kernels (including the 256-thread launch the stock RMS norm uses for 256 or more tokens). Off by default. |
| `MERLIN_COLD_EXPERTS_ADAPTIVE`, `MERLIN_COLD_EXPERTS_ADAPTIVE_PERIOD`, `_SWAPS`, `_WINDOW`, `_MARGIN`, `_MMAX` | engine (cold-expert offload) | **1** on `hy4flash-quad-3090` when the cold-expert offload is on, 0 elsewhere; 8 / 4 / 64 / 2 / 32 | `1` (with `MERLIN_COLD_EXPERTS`): every `PERIOD` steps each offloaded layer swaps up to `SWAPS` of its most-used cold experts into the GPU slots of its least-used hot ones, by byte copy outside the CUDA graphs. Placement only: serial outputs are unchanged; concurrent outputs vary run to run, as with any batch. Swap totals are logged when they change and at shutdown. |
| `MERLIN_EXPERT_HOT_COUNTS`, `MERLIN_EXPERT_HOT_WINDOWS`, `MERLIN_EXPERT_HOT_WINDOW_STEPS`, `MERLIN_EXPERT_HOT_MMAX`, `MERLIN_EXPERT_HOT_POLL_S` | engine (calibration) | off | Count routed-expert use per layer into per-rank JSON files, for building a cold-expert profile. Calibration only; does not change outputs. |
| `DFLY` | hy4flash launcher | **1** (DFly2 on) | `0` turns the DFly2 drafter off; only `0` or `1`. The measured 4-card pool was read with DFly2 on. |
| `COLD_STAGE` | hy4flash launcher | the last PP stage | With `MERLIN_COLD_EXPERTS` on, the PP stage the named cold-expert layers must sit on (`0` to PP - 1); ignored with a note when the offload is off. The KV pin is fixed per row, so the VRAM the offload frees is headroom and the KV pool does not change. |
| `ENGRAM` | hy4flash launcher | `fp8` | `fp8` serves the FP8 Engram pack, `bf16` the BF16 Engram pack. The checkpoint's `config.json` must agree (`engram_fp8_embedding` true = FP8; false or absent = BF16), else the launcher stops with both named. Never passed to the engine, which reads the tables' dtype from the checkpoint. |
| `--engram fp8\|bf16` | `./setup.sh --model hy4flash` | `fp8` | The Engram pack to plan, check the host for (host RAM, memlock, disk) and serve; it starts the launcher with `ENGRAM` set to match. Refused on every other model. |
| `SPEC_K` | hy4flash launcher | **3** (was 5) | DFly2 draft length on the hy4flash rows, set on `tools/merlin/launch.py`. `./setup.sh` starts a row at its default and drops a `SPEC_K` export. |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | hy4flash launcher | **64** on the hy4flash rows (the engine default is 512) | MiB of IndexPool scoring logits per slice. At 64 a prefill step past about 130K tokens of context is scored in more row slices, one extra scoring pass and all-gather per layer per slice, so the prefill memory stops growing with context. Bit-identical: the same per-token selection (slice bounds are equal on every DCP rank). An exported value wins. |
| `MERLIN_MOE_REPACK_CHUNK_MB` | engine (Marlin WNA16 MoE load) | 64 | Input MiB per expert chunk when post-load processing repacks a Marlin MoE layer in place, which bounds the load peak; `0` is the stock whole-layer repack. The packed weights are the stock ones. A whole number; anything else stops the start. |
| `MERLIN_MOE_CHUNK_TOKENS` | engine (Marlin MoE) | 0 (opt-in) | `N`: a step of more than `N` tokens runs the Marlin MoE call over `N`-token slices, with its caches sized for one slice (less workspace memory). Not bit-identical to the one-call path for steps above `N`; graphs at or below `N` tokens are stock; prefill cost unmeasured. `0` is the stock path. |

### Known limits

- **Hy4 Flash is new in this release.** The four-card row (`hy4flash-quad-3090`) is testing: its full-model runs
  passed on our 4 x RTX 3090 host at the 1,600,000,000 B pin (the same tokens as the stock kernels on serial prompts,
  262K prefill and decode with no out-of-memory, 13.19 ms per output token at K=3, a 609,398-token pool; the
  profile's evidence lists the decode and prefill rates), and kernel parity qualification is pending. The eight-card row (`hy4flash-octo-3090`, 4 x 1M) is untested on hardware. Hy4 Flash runs from `tools/merlin/launch.py` and `./setup.sh --model hy4flash`;
  `merlin serve` does not list it.
- **The KV RAM cache is not supported on Hy4 Flash yet.** On 4 x RTX 3090 its host copies were written but never
  restored, so it served no hits. `tools/merlin/launch.py` refuses `SWARM=1` and `MERLIN_KV_HOST_TIER_GIB` above 0 on
  every hy4flash row before the engine starts, and `./setup.sh` refuses `--kv-ram-cache` above 0 and `--swarm`.
- **The BF16 Engram pack is untested on hardware.** It needs about 198 GiB of host RAM available (118 GiB of pinned
  tables), more than our four-card host has; the Hy4 Flash figures above are the FP8 pack's.
- **Strict tool calls cost a little decode time at high concurrency** (1.03× at one stream, 1.00× at 8, 1.07× at 24
  on Flash-Next), the first request with a new bounded-string schema waits about 0.22 s longer than a repeat, and the
  first tool request after boot waits about 2.7 s longer (see Highlights). Strict stays opt-in.
- **Two value shapes in tool-call grammars are still slow, and the slow-fill guard ends them.** A string `pattern`
  that can contain `<` takes about 370 ms per mask fill, and a non-empty pattern (the `+` form) that can match spaces
  takes about 2 s per fill (measured on the CPU test host); the grammar writes both as xgrammar does (the `+` form would need thousands of states to stay
  exact). A request that keeps writing such a value reaches the guard's 5000 ms budget and ends with `Internal server
  error`; other requests continue. The guard is a backstop, not a fix.
- **DFlash2's cold prompts reach their first token about 18 to 19 % later** with the DFlash2 tick costs (see
  Highlights); `VLLM_MERLIN_PREFILL_RUNS_TICK_COSTS=139,25,1` restores the engine default.
- **The 27B `--swarm` KV RAM cache keeps its 48 GB default, and at that size it returns little on a busy trace.** On a
  144-session multi-turn agent trace on 4 × RTX 3090 (TP4, GPU pool 2,012,105 tokens), the server served from cache
  11.4 % of the trace's ceiling with the 48 GB host tier and 87.5 % with `--kv-ram-cache 96g`. At 96 GB the engine's processes peaked at
  about 121 GiB of host memory (about 73 GiB at 48 GB). These figures were measured on 2026-11-15, before the
  duplicates-first and waiting-last eviction orders became the defaults on 2026-11-18 (see Highlights).
- **With the KV RAM cache and the restore-start and seat-hold switches on, returning requests are mostly served from the host tier (opt-in).** The restore-start and seat-hold switches above (off by default; set each to `1`) with the per-slot credits (on by default) keep the queue in order: a return no longer jumps ahead while its prefix is still on the GPU, and by its turn that GPU copy has usually been reused, so it restores from host RAM. On a multi-turn agent trace on the 27B (TP4, 4 × RTX 3090, 96 GB KV RAM cache), returns were served 34 from GPU and 399 from the host tier (131 and 264 with the three switches off), and returns reached their first token sooner (p50 186 s against 260 s, p95 275 s against 358 s). The cost: measured alone on the 27B at 96 GB with the host tier on, `MERLIN_KV_HOST_TIER__RESTORE_START_ANY_STEP` costs about 1.9 percentage points of hit rate and adds unguarded losses, so it (and the seat hold, which needs it) stays off by default; unset or `0` is the earlier order. The per-slot credits alone cost nothing measurable and are on.
- **The 3.0.2.1 limits still apply** (the 20 GB rows untested on real cards, `tool_choice="auto"` and unfenced quoted
  markup, KV RAM cache restore waits), as do the 3.0.2 limits, except that inline code is now treated as code in
  tool-call markup (see Highlights).
- **The stable-residual repair is on by default** (it was off in 3.0.2.1). The failure it repairs (a rejected draft
  whose leftover distribution has no mass left in floating point) was not seen on Flash-Next or 27B TP4: 0 of 21,215
  rejections (14,132 and 7,083). With the same seeds, one request at a time, the repair changed no output token on
  either lane, and speed and the sampled GSM8K score stayed level. Other lanes (27B TP2 and TP1, ThinkingCap,
  Ornith, `SPEC=dflash2`) were not measured. `VLLM_SPEC_STABLE_RESIDUAL=0` restores the 3.0.2.1 residual.

### Docs

- Every user-facing 3.0.2.2 knob is in the table above; README Features links to it.
- SETUP gains the update steps for a checkout of a release tag and lists every folder setup writes under Uninstall;
  the README's manual install pins the checkpoint revision.
- The 27B README and reference state the KV RAM cache hit rates at 48 GB and 96 GB, the 27B reference the four-card
  P2P settings, and the Flash-Next reference the DFlash2 tick costs and the KV RAM cache lock timeout.
- A community bench card for Qwen3.8-27B on two modded RTX 3080 20 GB cards with dynamic-BAR1 P2P (2026-11-15,
  contributed by brandon-nuttall; the contributor's measurements, not the maintainers') is added, and the
  `27b-tp2-sm86-20g` row's P2P note and evidence cite it. The card and the note say that the dynamic-BAR1 driver
  (`noon-at-cgn/geforce-dynbar1-p2p`) is a patched third-party stack that Merlin does not ship or test, and that P2P
  stays off on the static-BAR1 595-era stack. The row stays untested by Merlin.
- The README credits link every handle; two names stay in NOTICE only.

## Merlin 3.0.2.1

2026-11-12. Built on vLLM 0.33.0, like 3.0.2, with the same seven models. This release fixes the KV RAM
cache slowdown on a full cache, ships the 4 × 20 GB Flash-Next family rows and changes five serving defaults: shared-prefix
checkpoints and prefill runs on Flash-Next, quoted tool calls kept as text on the Qwen3 parser, the
two-card 27B P2P levers, and Super VL's earlier turns keeping their thinking in the prompt.

### Highlights

- **The KV RAM cache no longer slows decode once it is full** (halt95/merlin-engine#11, reported by
  [jbeemer1](https://github.com/jbeemer1)).
  - **The cause:** once the host cache was full, the write-back planner rebuilt its list of eviction candidates on
    every step. One-stream decode fell from about 120 tok/s on a fresh cache to 72.6 tok/s on a full one, with the
    planner taking 78 % of the engine's step time.

  - **The fix:** the planner now keeps that list from step to step and re-derives only what changed. It picks the same
    blocks to evict as 3.0.2, with one exception: a stale pin count that could misclassify a block is fixed.
  - **Results:**
    - full cache, one stream: 72.6 → 122.3 tok/s, with the planner at 18.4 % (a fresh cache does about 120);

    - full cache, eight streams: 389–460 → 596 tok/s (two stock reference runs);

    - output tokens identical;
    - on a realistic multi-turn trace the hit rate stays at 97.4 % of what the trace can reuse, 0.25 points below
      3.0.2 and inside the run-to-run band.

    These figures come from the reporter's stress probe on 4 × RTX 3090 (thinking off, temperature 0, three
    fixed-length 800-token sends, steady-window tok/s), not the bench-card protocol. On ordinary chat output with
    thinking on, see the bench-card decode figures in the table below.
- **More KV RAM cache fixes.** None of these changes output tokens.
  - **A cancelled restore cleans up.** Before, a request cancelled while its restore waited between copy batches (or
    for its first batch) kept its GPU blocks until the server restarted, and an engine reset at that moment left the
    request waiting for good. Now an aborted request frees its blocks and a reset request recomputes. The host copies
    it read are kept.
  - **Host memory is fully reused.** Two cases left host rows that no conversation owned unavailable until a reset:
    two copies of the same block, and a row partly replaced by a newer turn. Both rows are now reclaimed.
  - **A conversation's previous turn is released once the next turn is stored,** also when the turn boundary falls
    inside an attention page. The turn just before the newest one is kept, so a retry, a regenerate or a fork of the last
    turn still restores from the cache.
  - **Admitting a new request beside a host restore no longer evicts a block that a waiting conversation needs.** The
    eviction forecast now counts the blocks the restore itself takes. This case was not seen on the shipped Flash-Next
    launch in a realistic soak.
  - **A returning conversation's restore keeps the GPU blocks it needs.** While its prefix is copied back from host
    RAM, new requests no longer take the blocks it still needs to finish its prefill. Ordinary running requests reserve
    nothing for it and are not preempted or slowed for it. A session on the streaming-input API (no shipped launcher
    uses it) leaves the restore the same room when it grows, and preempts itself if that room is all it lacks. If a restore still cannot get its blocks while nothing else runs, it goes back to the
    prefix cache and is admitted again, with no work lost, instead of waiting for good.
  - **With routed-expert capture on,** a host restore stops at the captured history, as a GPU cache hit does.
  - **Long runs with many preemptions** no longer grow a per-request table without bound.
  - **With the optional per-rank copy byte cap** (`MERLIN_KV_HOST_TIER__COPY_BYTES_D_MIB`, off by default), one
    write-back pass stays within the cap.

  Every fix above except the returning-restore one was measured over six realistic multi-turn soaks against the same
  build without them: hit rate, churn and speed are within their bands, output tokens are identical, and cached
  conversation checkpoints lost per million prompt tokens fall from about 4.4 to about 3.8.

  The returning-restore fix was soaked on its own against the previous development build over three pooled runs on the shipped swarm
  launch: hit rate (89.4 % against 89.4 %), cached conversation checkpoints lost per million prompt tokens
  (3.7 against 4.3) and output tokens were unchanged within the run-to-run band, with no reservation
  preemption or forced release recorded.
- **Flash-Next: shared-prefix checkpoints (FGPC) are on.** Sessions often open behind the same system prompt. When that
  shared prefix covers at least one 4,096-token attention block, a new session's first turn now resumes from the cached
  junction instead of recomputing the prompt.
  - **Measured on the shipped swarm launch with a ~9K shared system prompt:**
    - first turns served from the cache: 0 of 82 with FGPC off, 43 of 87 and 41 of 85 in two runs with it on;

    - speed, a long soak and output tokens are within their run-to-run bands;
    - retrieval from a junction matches a cold recompute (48 of 48 needles on both), and its output drift is no larger
      than a plain prefix-cache hit's.

  - **Where it applies:** Swift1.5 and Swift1.5-Huihui use the same launcher and get it too. The 27B family already ran
    FGPC.
  - **Turning it off:** `merlin serve <model> -- --no-enable-mamba-shared-prefix-checkpoint`.
  - **Junction ageing:** junction blocks now age like any other cached block. Before, a free junction was the first
    block taken, so under a full pool the next session often found it gone.
- **Flash-Next: a long cold prompt now takes turns with the running sessions** (halt95/merlin-engine#6 item 1, reported
  by Sudzpel). Instead of sharing every step with them, the prompt prefills in runs of 16 full 1,024-token steps, and
  between runs the sessions decode alone. How long they decode alone is a configured scheduling target (decode share
  0.30), counted in decode rounds, not a measured share of time.
  - **Measured on the reference box (4 × RTX 3090) in four runs of the shipped setting** (decode share 0.30: one on the
    first build of this change, three with the shipped default launch line; one more run at share 0.25, not the shipped
    setting, is not counted), **with three sessions decoding through a cold prefill:**
    - those sessions keep 31–38 % of their decode speed during a cold 64K–128K prefill (36–37 % at 64K, 31–38 % at
      128K), up from 14–16 % with 512-token slices alone;
    - a session that starts decoding during the prefill: first token 0.65–0.69 s, with a free session slot (0.69–0.78 s
      with slices alone);
    - the cold prompt's own first token comes later: +17 to +19 % at 64K, +12 to +14 % at 128K, +19 to +24 % at 262K;
    - the p99 gap between streamed tokens is 0.40 s in three of the four runs and 0.58 s in one where the engine showed
      no slow step (a client-side reading), against 0.23–0.35 s without the runs.
  - **Measured in the same four runs with no session decoding:**
    - three 64K prompts sent together: first token p95 34.5–34.7 s (28.3–28.9 s with slices alone);
    - a returning turn's first token, with no session decoding when it arrives: 1.0–1.2 s against 1.1–1.3 s without
      the runs once the kernels were cached; in the other two runs the returning turns met the compile pause under
      Known limits (one step stalled about 2.6–2.7 s) and the row read about 2.3 s (2.29 and 2.31 s); the previous
      build's first boot met the same pause;
    - lone prompts and prefill: within the run-to-run spread of about 1 %;
    - a single session's output is token-identical.
  - **Memory:** lowest free GPU memory over a run 943–957 MiB, against 959–971 MiB with slices alone.
  - **Where it applies:** the Flash-Next family launcher rows only (Swift1.5 and Swift1.5-Huihui use the same launcher),
    with `--spec dflash2` too on the 4 × 24 GB rows (see Fixes for its KV pin). The launcher raises `--max-num-batched-tokens` by one verify row per session slot (1,056 on four 24 GB cards), so the
    sessions never shrink a prefill step. The 27B, Super VL and Ornith launchers and a direct `vllm serve` are unchanged.
  - **Turning it off:** `PREFILL_RUNS=0` restores the plan without it (512-token slices, budget 1,024 on four 24 GB
    cards). `LPT=<n>` still sets the slice size used while no session is decoding. When `CHUNK` and the block size
    (`BLOCK_SIZE`) do not tile (neither divides the other) or `BLOCK_SIZE` is below the hybrid-state floor, the default
    switches the runs off with a note at start-up; an explicit `PREFILL_RUNS=1` refuses to start instead.
  - **Not covered:** with every session slot in use, a new prompt still waits for a slot (`SEQS`, the launcher's
    `--max-num-seqs`; 8 on four 24 GB cards).
  - `VLLM_MERLIN_DECODE_FIRST=1` (experimental) cannot be combined with this default: the launcher refuses it unless
    `PREFILL_RUNS=0`.
- **Flash-Next on four 20 GB cards (profiles included, not yet tested on real cards ⚫).** Two new rows:
  - **The rows:** `fn-quad-20g-sm86` for the RTX 3080 20GB and RTX A4500, and `fn-quad-20g-sm89` for the RTX 4000 Ada.
    Each holds one 262K session.
  - **Static cold-expert offload, on by default on these rows:**
    - on MoE layers 11, 12, 19, 20 and 37, the most-used half of each card's experts stays on the GPU;
    - the rest is kept in pinned host RAM and read over PCIe;
    - placement only, output identical.
  - **Turning it off:** `MERLIN_COLD_EXPERTS=0`. The 20 GB fit was measured only with it on.
  - **Host RAM:** the offload pins about 8 GiB of extra host RAM, which preflight's RAM check does not count.
  - **ECC:** turn it off on the RTX A4500 and RTX 4000 Ada.
  - **Evidence:** emulation only (4 × RTX 3090, each with 4 GiB held back), five emulated boots
    on 2026-11-07 and 2026-11-09, outputs identical to the earlier 20 GB run (four identity prompts).
  - **Swift1.5 and Swift1.5-Huihui** get the same two rows (`fn-swift1.5-quad-20g-sm86`, `fn-swift1.5-quad-20g-sm89`, `fn-swift1.5-huihui-quad-20g-sm86`, `fn-swift1.5-huihui-quad-20g-sm89`): same layout, pin, env and
    cold-expert profile (the three checkpoints share one config.json); the evidence is the Flash-Next emulation, and
    neither fine-tune was booted at 20 GB.

  The Qwen3.8-27B family gains 20 GB rows too, also untested ⚫: TP4 (`27b-tp4-sm86-20g`, `27b-tp4-sm89-20g`) and TP2
  (`27b-tp2-sm86-20g`, `27b-tp2-sm89-20g`). Preflight and `merlin serve --dry-run` accept `--simulate 4x3080-20g` and
  `--simulate 4x4000ada` (`2x...` for the TP2 rows).
- **Qwen3 parser: quoted tool calls stay text (on by default).** This applies to the `qwen3` parser and the parsers built
  on it, which the Flash-Next family, the 27B family and Ornith 1.5 serve with.
  - **Code blocks:** a tool call inside a fenced code block is returned as text, not as a real call.
  - **Quoted reasoning:** a tool block quoted inside reasoning that the model then closes is returned as text too.
  - **Unoffered names:** a call to a tool the request did not offer is kept as text in the field the model wrote it in,
    and it no longer ends reasoning.

  `MERLIN_QWEN3_LITERAL_SPANS=0` turns the quoting rules off (the 3.0.2 behaviour). This narrows the 3.0.2 known limit
  (see Known limits).
- **Qwen3.8-27B on two RTX 3090s with P2P on** (`27b-tp2-3090`) sets six launcher defaults: `NCCL_PROTO=Simple`,
  `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, `VLLM_DRAFT_BOUNDED_PLAN=1`, `VLLM_GDN_DECODE_FASTPATH=1`,
  `VLLM_GDN_SHARED_ALIGN_INDEX=1` and `VLLM_TARGET_SEQ_LENS_ASYNC=1`.
  - **Results** (an A/B of these defaults, not a bench-card row; the card's figures against 3.0.2 are below): one-stream decode +16.0 % (pooled, against a 7.2 % run-to-run band); eight streams within the
    run-to-run band; outputs identical on all 28 identity checks.

  - **ThinkingCap's two-card launcher** uses the same profile, so it gets these defaults too (on 3.0.2.1 its decode is +19 % / +6 % at 4K / 131K against 3.0.2, and level at 262K (step time 29.26 ms against 29.34 ms; the 128-token rate varies with the text sent, 12 sends on the published wheel); see its bench card).
  - **Unchanged:** the other 27B rows, and this row with `P2P=off`.
- **Qwen3.8-27B captures CUDA graphs for its widest decode steps.**
  - **Four cards** (`27b-tp4-3090`; ThinkingCap's four-card launcher uses the same row): decode sizes 80 and 120, the
    steps at 16 and 24 streams. With the new graphs in use, 16 streams decode at 1,218.3 tok/s and 24 streams at
    1,332.6 tok/s (ThinkingCap: 1,202.7 and 1,321.5 tok/s). Outputs at 16 and 24 streams were compared with the previous capture list over repeated runs and
    showed no difference distinguishable from run-to-run variation; byte-identical output under concurrency is not
    claimed. The 8 MP image floor is 796 MiB on every card, counted as total minus used and sampled every 100 ms on a fresh boot with no decode before the images (the 27B checkpoint and ThinkingCap, one run each); with decode run first, the readings at one-second sampling were 798 and 796 MiB.
  - **Unchanged:** the other four-card rows keep the shipped list.
  - **Two cards** (`27b-tp2-3090`; ThinkingCap's two-card launcher uses the same row): decode size 80, the step at 16
    streams. The 8 MP image floor is 764 MiB, counted as total minus used. The other two-card rows keep the shipped
    list.
- **Nemotron 3.5 Super VL keeps earlier turns' thinking in the prompt by default.** Its chat template used to drop the
  thinking of earlier answers at every new user turn. The prompt then forked inside the previous answer, and the prefix
  cache missed from there.
  - **The change:** the launcher now passes `{"truncate_history_thinking": false}`, so returning turns hit the cache.
  - **The cost:** the returning prompt is longer (thinking included), so its first token comes later: median 0.348 s
    against 0.313 s. Returning turns are served from cache more often (80.1–80.7 % against 78.4–78.7 %), and decode is
    unchanged within run-to-run noise.

  - **Opting out:** a request's own `chat_template_kwargs` can still set `truncate_history_thinking: true`.

### Measured on Merlin 3.0.2.1

4× RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W; runs of 2026-11-10 to 2026-11-12 (UTC). The Flash-Next 4× 24 GB, Swift1.5 and Swift1.5-Huihui cells were measured on the published wheel on 2026-11-11, except two thinking-off 8-agent cells; ThinkingCap's two-card 262K decode was re-measured on it on 2026-11-12 (12 sends); the other cells were not re-measured on it and come from the previous build of that wheel (built on 2026-11-10, before the prefill runs change), which does not have the new prefill schedule (on by default on the Flash-Next 20 GB row too, and not measured there); Super VL's thinking-off ladder on its card ran on an earlier build (2026-11-09). Columns as in 3.0.2:
- decode is one stream, thinking on (effort medium where the template has levels), median of 3 (ThinkingCap 2× at 262K: median of 12);
- 8 agents is the total of eight simultaneous follow-ups on 16K conversations.

The 20 GB row is measured on 4 × RTX 3090 that emulate 20 GB cards (4 GiB held back per card), not on real 20 GB cards.
The Flash-Next and Swift1.5-Huihui thinking-off 8-agent cells are from the previous build: their runs on the published wheel ended early (one reply stopped by itself; overlap windows of 0.25 and 0.29 s) and were not used.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next | 4× 24 GB | 924,993 | 174 / 188 / 192 | 6,839 tok/s | 640 / 641 | 197 / 198 |
| Flash-Next | 4× 20 GB (emulated) | 265,888 | 147 / 158 / 165 | 4,555 tok/s | n/a (pool holds 7 of 8) | n/a |
| Qwen3.8-27B | 2× | 826,689 | 248 / 197 / 160 | 1,117 tok/s | 740 / 677 | - / 194 |
| Qwen3.8-27B | 4× | 2,012,105 | 262 / 258 / 233 | 2,115 tok/s | 970 / 857 | 198 / 195 |
| Swift1.5 | 4× | 924,993 | 180 / 185 / 182 | 6,793 tok/s | 688 / 661 | 199 / 198 |
| Swift1.5-Huihui | 4× | 924,993 | 178 / 190 / 203 | 6,893 tok/s | 719 / 668 | 198 / 197 |
| Qwen3.8-27B | 1× | 271,506 | 131 / 102 / 75 | 587 tok/s | n/a (two at most) | - / 194 |
| ThinkingCap | 2× | 826,689 | 244 / 179 / 163 | 1,116 tok/s | 731 / 663 | - / 195 |
| ThinkingCap | 4× | 2,012,105 | 240 / 238 / 233 | 2,115 tok/s | 974 / 856 | 196 / 193 |
| Nemotron 3.5 Super VL | 4× | 1,856,367 | 105 / 102 / 101 | 4,193 tok/s | 444 / 417 | 196 / 194 |
| Ornith 1.5 | 4× | 894,618 | 149 / 122 / 107 | 2,284 tok/s | 516 / 484 | 195 / 191 |

The 20 GB row has no GSM8K run; the one- and two-card rows have no thinking-on GSM8K-200 run (`-`), as in 3.0.2, because it runs on the four-card boot.

Against 3.0.2:
- **Flash-Next 4× 24 GB:** Thinking on, decode +1 % / 0 % at 4K / 131K, within run-to-run spread, and 191.8 tok/s at 262K (−7 %; median of 3 sends, 191.4–200.3); thinking off, 143.6 tok/s at 262K (−6 %; 142.6–147.4); 262K prefill +1 %; eight agents −2 % in one run of 8, which is no gain or loss: the same row gave 661–697 tok/s (mean 680) over six repeated runs.
- **Qwen3.8-27B 2×:** Thinking on, decode +18 % / +7 % at 4K / 131K and 159.9 tok/s at 262K (+10 %; median of 3 sends, 153.1–165.9); thinking off, 105.1 tok/s at 262K (−2 %; 96.1–118.6); 262K prefill 0 %; eight agents +9 % in one run of 8, a short overlap window, so read it loosely (the six P2P launcher defaults above).
- **Qwen3.8-27B 4×:** Thinking on, decode +2 % / +1 % at 4K / 131K, within run-to-run spread, and 232.7 tok/s at 262K (−7 %; median of 3 sends, 230.7–252.6); thinking off, 163.3 tok/s at 262K (0 %; 163.0–165.9); 262K prefill 0 %; eight agents +3 % in one run of 8, a short overlap window, so read it loosely.
- **Qwen3.8-27B 1×:** 262K decode −2 % thinking on, +1 % off.
- **ThinkingCap 2×:** 262K decode level on step time (29.26 against 29.34 ms on, 28.72 against 28.73 ms off; 12 sends on the published wheel).
- **ThinkingCap 4×:** 262K decode +4 % thinking on, −1 % off.
- **Swift1.5:** 262K decode −2 % thinking on, +11 % off.
- **Swift1.5-Huihui:** 262K decode +5 % thinking on, −5 % off.
- **Nemotron 3.5 Super VL:** 262K decode −4 % thinking on, +9 % off.
- **Ornith 1.5:** 262K decode −4 % thinking on, +6 % off.
- **Exact-262K decode:** 191.6 tok/s (172.4–200.3) on Flash-Next and 238.1 tok/s (230.7–252.6) on the 27B 4× (median and
  range over 6 sends). These rows generate 128 tokens from a 261,888-token prompt, one stream, and each send uses a different prompt
  text. With speculative decoding the number of tokens produced per decode step depends on the text being generated, so at
  128 tokens the rate varies from send to send with the text; the bench cards give the range beside each median. Measured on
  this release on the 27B two-card layout with thinking off: the same text sent 100 times, across four server starts, gave
  identical output and 100.0–103.5 tok/s (2.783 tokens per step on every send); three other texts gave 106.9–110.3 tok/s;
  the time per decode step was flat, 27.5–27.9 ms at the client and 28.3–28.6 ms in the server. On that row the previous
  development build and this release gave identical output and the same rate (a paired difference of +0.3 to +0.5 %, inside
  the difference of about 1 % between two starts of the same build). That was measured on that one layout and thinking mode.
  On Flash-Next each decode step at exact 262K is within 2 % of 3.0.2; the published 3.0.2 figure of 207 tok/s came from
  different send texts (a repeat run of Flash-Next on 3.0.2 measured 193 tok/s, sends 184–207).

Every row is in the bench cards: [Flash-Next](../models/flash-next/benchmarks/2026-11-12/BENCH-CARD.md) · [Swift1.5](../models/swift1.5/benchmarks/2026-11-12/BENCH-CARD.md) · [Swift1.5-Huihui](../models/swift1.5-huihui/benchmarks/2026-11-12/BENCH-CARD.md) · [Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-12/BENCH-CARD.md) · [ThinkingCap](../models/thinkingcap/benchmarks/2026-11-12/BENCH-CARD.md) · [Nemotron 3.5 Super VL](../models/nemotron-super-vl/benchmarks/2026-11-12/BENCH-CARD.md) · [Ornith 1.5](../models/ornith1.5/benchmarks/2026-11-12/BENCH-CARD.md).

### Before you upgrade

- **New wheel.** The wheel is `vllm-0.33.0+merlin.c56eeed392-cp313-cp313-linux_x86_64.whl`; `release/merlin-wheel.sha256` names it, so this checkout refuses the 3.0.2
  wheel. The dependency lock (`release/requirements-0.33.lock`, 223 pins) is unchanged. On the reference host, with the published wheel, Swift1.5 booted in 432 s and Swift1.5-Huihui in 412 s, each the first start of its configuration (graphs compiled, none loaded from cache). On the previous build, a later start of the four-card Flash-Next layout that loaded the cached compiled graphs was ready in 166 s against 301 s for its first start; the saving is the compile cache and graph warm-up, and weight loading was not faster (most weights were read from disk in both starts).
- **New defaults you may notice:**
  - Flash-Next family: FGPC on, `--long-prefill-token-threshold 512`, and the prefill runs, which raise
    `--max-num-batched-tokens` to 1,056 on four 24 GB cards; see Highlights for how to turn each off.
  - Qwen3 parser: fenced and quoted tool calls are returned as text; `MERLIN_QWEN3_LITERAL_SPANS=0` restores 3.0.2.
  - Super VL: earlier turns keep their thinking in the prompt.
  - 27B on two cards with `P2P=on`: an inherited `NCCL_PROTO` or `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` is not
    used (the P2P mode sets them). Your own values of the other four switches are kept.
- **Flash-Next launchers pass two more `MERLIN_` variables through.** These launchers clear every inherited `MERLIN_`
  export except the ones they own.
  - `MERLIN_QWEN3_LITERAL_SPANS` and `MERLIN_KV_TIER_DIAG` now reach the server. Any value other than `0` or `1` stops
    the launch.
  - `MERLIN_REPACK_RELEASE` and `MERLIN_LOAD_PEAK_LOG` are still cleared there, so the engine defaults apply.
- **`KV_BYTES=auto` on the 27B** keeps the CUDA-graph memory estimate on, and the automatic KV size now leaves room for
  what the engine allocates beside the KV cache. An automatic pool can be slightly smaller than in 3.0.2. Pinned rows (the
  default) are unchanged.

### Fixes

- **DFlash2 KV pin 3,468,165,120 → 3,630,000,000 bytes, pool 789,668 → 815,203 tokens.** The prefill runs' step
  budget costs the DFlash2 drafter's sliding-window layers one block per request, which took the pool below
  3 × 262K. At the raised pin (FP32 GDN state, align-mode prefix caching) the engine fits three 262K sessions as
  before (five 131K), and the prefill runs are on with DFlash2 as on the rest of the family.
- **Flash-Next on 20 GB cards starts reliably:**
  - the pipeline-parallel and PCIe-IPC communicators are created before memory profiling, so the KV sizing sees
    their memory;
  - the allocator's cache is freed after each module's weight repack (`MERLIN_REPACK_RELEASE`, on by default);
  - the QSA top-k index buffer is shared across a stage's layers.

  None changes numerics.
- **Weight offload with Marlin MoE checkpoints works again on 0.33.** With `--cpu-offload-gb` / `--cpu-offload-params`,
  the repacked expert weights stayed on the GPU layer after layer, and the boot ran out of memory (from
  vllm#62326 in the 0.33 base). Offloaded weights now go back to host RAM.
- **`--cpu-offload-params` can name a layer.** A pattern such as `layers.3.mlp.experts` now selects layer 3 only (not
  layer 30), on the card that builds it. Existing patterns match as before.
- **Flash-Next decode top-k reads only valid positions** (port of vllm#62467). On a row whose length is not a multiple
  of 4 and that has fewer than top-k finite scores, the selection could read past the row and return an out-of-range
  index.

- **`./setup.sh` reads a four-part release heading.** The release version comes from the first `## Merlin`
  heading in this changelog, and it was cut to three parts. A four-part heading therefore gave the wrong wheel download
  URL, and the wrong version in the `./setup.sh` and `merlin pull` banners. `MERLIN_RELEASE_VERSION` still overrides it.
- **The compile-cache guard no longer stalls other ranks when it cannot set a cache aside.** If moving a faulted
  compile cache aside failed, or the check was interrupted, the process kept an exclusive lock on that cache directory
  for its whole life, so the other ranks and any other engine using the directory waited at start with no end. A failed
  move is now served as before, with the same warning, and the other ranks start; an interrupted check runs again on
  the next try.
- **No "unknown variable" warning for documented tool variables.** `MERLIN_READY_BANNER`, `MERLIN_PROFILE`,
  `MERLIN_ALLOW_SERVER_DEV_MODE` and `MERLIN_EMULATE_VRAM_CLASS` are read only by the Merlin tools, but every boot that
  followed the docs warned that they might be misspelt.
- **KV RAM cache:** see Highlights. The restore-stall warning in the tier census now fires only while restores are
  wanted; before, a cache that served every return from the GPU could trip it.

### Engine

- **Prefill runs** (`VLLM_MERLIN_PREFILL_RUNS`, off in the engine, on from the Flash-Next launcher): while a session
  decodes, a long prefill advances in runs of full chunks and then waits for a decode-only window; a short prefill (a
  new turn, a session joining late) takes its steps first while a session slot is free. The windows are counted in
  decode rounds, not timed: the runs' decisions depend only on the scheduler-visible sequence of arrivals, completions
  and cache state, not on timers. One log line per 1,000 steps counts the windows and runs.
- **New debug lines:** `MERLIN_KV_TIER_DIAG=1` logs shared-prefix junction events, one line per first-turn admission and
  one line per host eviction. Off by default; output is unchanged.
- **Boot memory report:** `MERLIN_LOAD_PEAK_LOG=1` logs per-module memory peaks for model construction, weight loading
  and post-load processing. Off by default.
- **Profile rows can be marked blocked:** preflight then refuses them with the reason, and no launcher is rendered.
  No row ships blocked.
- **Tier counters:** the KV RAM cache counters line gains `units_lost_covered_early` (evicted kept predecessors among
  `units_lost`), and the tier metrics line gains `restore_demand` and `restores_pending`; the plan counters gain `restore_refused`,
  `restore_refused_evicted`, `reservation_preemptions`, `reservation_preempt_tokens` and `restore_released`
  (parked-restore reservations).

### New flags and environment variables

| Name | Where | Default | What it does |
|---|---|---|---|
| `--enable-mamba-shared-prefix-checkpoint` (FGPC) | Flash-Next family launcher | **on** (was off) | Sibling first turns resume from the shared-prefix junction. `-- --no-enable-mamba-shared-prefix-checkpoint` turns it off. |
| `LPT=<n>` (`--long-prefill-token-threshold`) | Flash-Next family launcher | **512** (was off) | Prefill slice size for long prompts while no session is decoding (the prefill runs take over while one is); `LPT=0` turns slicing off. |
| `PREFILL_RUNS` | Flash-Next family launcher | **1** (new) | A long cold prompt takes turns with the running sessions (see Highlights). `0` turns it off and restores the step budget to `CHUNK`. Blank is unset (on). The launcher steps it aside, with a note, when `CHUNK` and `BLOCK_SIZE` do not tile (neither divides the other) or `BLOCK_SIZE` is below the hybrid-state floor; `PREFILL_RUNS=1` refuses those instead. |
| `VLLM_MERLIN_PREFILL_RUNS` | engine | off; on from the Flash-Next launcher | The engine switch behind `PREFILL_RUNS`; with it on, the engine refuses `VLLM_MERLIN_DECODE_FIRST=1`, `--prefill-schedule-interval` above 1, a step budget (`--max-num-batched-tokens` or `--max-num-scheduled-tokens`) without one verify row per session slot, a chunk or a nonzero `--long-prefill-token-threshold` below 256 tokens, and a chunk below the drafter's prefill lookahead. The launcher checks the same after your extra flags and names the knob to change. |
| `VLLM_MERLIN_DECODE_FIRST=1` | engine | off | Still experimental (see 3.0.2); cannot be combined with the prefill runs: on a Flash-Next row set `PREFILL_RUNS=0` to use it. |
| `MERLIN_QWEN3_LITERAL_SPANS` | engine (`qwen3` parser); reaches the server from every launcher | on | `0` turns the fenced-span and quoted-reasoning rules off (3.0.2 behaviour); the unoffered-name rule stays on. The Flash-Next launchers accept only `0` or `1`. |
| `MERLIN_COLD_EXPERTS=<profile.json>` | engine; checked by every launcher | off; on from the row on the 4 × 20 GB rows | Static cold-expert offload. Unset, blank or `0` is off everywhere, direct engine use included. A path replaces the row's profile. Every launcher (Flash-Next, 27B, Super VL, Ornith) checks a profile before the boot, including that it matches the checkpoint's `config.json`; on a lane other than Flash-Next, a profile that fits is noted as not measured there. A profile not tied to a checkpoint, or one checked against a checkpoint with no local `config.json`, is allowed with a warning. |
| `MERLIN_COLD_EXPERTS_LAYERS` | as above | the profile's | Comma-separated decoder layers to offload. Ignored, with a launcher note and an engine warning, when `MERLIN_COLD_EXPERTS` is off. |
| `MERLIN_COLD_EXPERTS_HOT_FRAC` | as above | the profile's | Share of each layer's experts kept on the GPU, from 0 up to but not including 1 (0 = the whole layer offloaded). Ignored, with a launcher note and an engine warning, when `MERLIN_COLD_EXPERTS` is off. |
| `REASONING_PARSER`, `TOOL_CALL_PARSER` | every launcher | the recipe's parsers | A vLLM parser name replaces the recipe's (`hf` reads the checkpoint's own `response_template`). `none` turns the parser off: the launcher leaves out the recipe's `--reasoning-parser`, or its `--tool-call-parser` and `--enable-auto-tool-choice`, and says so. An empty value is unset, with a note. |
| `GENERATION_OVERRIDE` | Flash-Next, 27B and Ornith launchers | the recipe's sampling | A JSON object for `--override-generation-config`. `none` leaves out the recipe's `--override-generation-config`, with a note when the recipe passes one. An empty value is unset, with a note. |
| `USE_CKPT_TEMPLATE` | Flash-Next and 27B launchers | on for a checkpoint of a newer model generation that the launcher does not recognise by name and that ships a chat template, with a note; off otherwise | `1`: the checkpoint's own chat template and sampling; `0`: the recipe's; blank is unset. |
| `MERLIN_REPACK_RELEASE` | engine | on | Frees the allocator's cache after each module's weight repack when at least 256 MiB is cached. `0` = stock vLLM. Cleared by the Flash-Next launchers. |
| `MERLIN_LOAD_PEAK_LOG=1` | engine | off | Logs the memory peaks of model construction, weight loading and each module's post-load processing. Cleared by the Flash-Next launchers. |
| `MERLIN_KV_TIER_DIAG=1` | engine; passed through by every launcher | off | Debug log lines for shared-prefix junctions, first turns and host evictions. Only `0` or `1`. |
| `MERLIN_EMULATE_VRAM_CLASS=<GiB>` | `merlin serve` | off | **Test only.** Matches every card as a smaller VRAM class (for example `20`), with an `EMULATION:` note. It only goes down, and it never reaches the engine. |
| `--simulate 4x3080-20g`, `--simulate 4x4000ada` | preflight, `merlin serve --dry-run` | n/a | Plan for RTX 3080 20GB or RTX 4000 Ada cards without them. |
| `--cpu-offload-params layers.N...` | vLLM | n/a | A pattern can now name one layer (see Fixes). |
| `VLLM_DRAFT_BOUNDED_PLAN`, `VLLM_GDN_DECODE_FASTPATH`, `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_TARGET_SEQ_LENS_ASYNC` | 27B launcher, `27b-tp2-3090` with `P2P=on` | **1** (was unset) | Two-card P2P decode levers. Your own value is kept. |

### Known limits

- **The 20 GB rows are untested on real 20 GB cards** ⚫. Their evidence is the emulation on 24 GB cards.
  - **Flash-Next:** one 262K session; eight agents do not fit the pool (it holds seven of the eight 16K follow-ups; two lost their cached prefix). The emulated `fn-quad-20g-sm86` row fits
    with at least 128 MiB reported free by nvidia-smi (`memory.free`) on every card, 8 MP images included. Over the 8 MP image rows the lowest free memory,
    counted as total minus used and sampled every 100 ms, was 955 MiB on two cards and 903 MiB on the other two on the previous build (2026-11-10; not re-measured on the published wheel), and 835 and 863 MiB in an earlier run on the previous development build: 835 to 955 MiB across the two runs, all above the 24 GB rows' 524 MiB margin.
  - **The 27B 20 GB rows** have estimated capacity only.
  - **Decode will be slower than the emulation shows** on cards with less memory bandwidth (RTX A4500, RTX 4000 Ada).
- **The Flash-Next prefill runs cost the cold prompt and the gaps between streamed tokens:**
  - **The cold prompt's own first token** comes later when sessions are running: +17 to +19 % at 64K, +12 to +14 % at
    128K, +19 to +24 % at 262K.
  - **The p99 gap between streamed tokens** is 0.40 s in three of the four runs and 0.58 s in one (a client-side
    reading; the engine showed no slow step), against 0.23–0.35 s with 512-token slices alone.
  - **Three cold 64K prompts sent together** with no session decoding reach their first token (p95) in 34.5–34.7 s,
    against 28.3–28.9 s with slices alone.
  - **A new prompt still waits for a session slot** when every slot is in use (`SEQS`, the launcher's `--max-num-seqs`).
  - **Measured on Flash-Next in four runs of the shipped setting on the reference box** (4 × RTX 3090) with three
    sessions decoding through the cold prefill; the Swift1.5 and Swift1.5-Huihui rows use the same settings and were
    not measured for this.

  `PREFILL_RUNS=0` turns the runs off.
- **After a fresh install, the first step that mixes several short prefills with decode rows can pause once for 2–3 s**
  (for example returning turns whose first tokens are being decoded)
  while its kernels compile (on the published wheel and on the previous build alike); it does not recur once the cache
  holds them.
- **The eight-agent rows are single runs** and move by several percent between identical runs (661–697 tok/s across six
  repetitions on Flash-Next). A thinking-off eight-agent row is void when any one reply ends early; that happened on
  at least one model in each run of this release, and those rows keep a figure from another run.
- **Engine: a prompt that ends within the speculative lookahead of a block boundary can stall** with a draft head of
  more than one prediction layer and more than one speculative token. Every checkpoint in this release has a one-layer head, so it cannot occur on the shipped models; 3.0.2 has it too.

- **`tool_choice="auto"` can still return quoted tool-call markup as a real call** when the quote is not fenced and is
  outside reasoning. An example block naming an offered tool, followed by the real call, is still read as a call. Send
  `tool_choice="none"` for explain-only requests. The other 3.0.2 parser limits still apply.
- **Inline code is not yet treated as code in tool-call markup.** On the Qwen3 parser, a tool call quoted in a fenced
  code block (a line that starts with three or more backticks or tildes) stays text. A call quoted in inline code
  (single backticks inside a sentence) is still read as markup.
- **Restores from the KV RAM cache can still wait, and a new request can wait for one.** Neither loses work.
  - A returning conversation's restore can still wait behind requests that were already running when it landed.
    Ordinary running requests are not preempted for it (a growing streaming-input session can preempt itself), and
    requests that arrive meanwhile queue behind it.
  - A newly arrived request that fits the free blocks, but not beside a restore that is landing, waits for that
    restore: about one restore copy, slightly longer on lanes that keep recurrent-state checkpoints.
  - The parked-session stall on the streaming-input API (a session whose input arrives in chunks, used by realtime
    speech-to-text and by no shipped launcher) is unchanged in this release.
- **The stable-residual repair for speculative decoding is off by default.** With sampling (temperature above 0), the
  rejection sampler computes the leftover distribution in plain floating point, which can lose precision when the
  draft and the target model agree closely. `VLLM_SPEC_STABLE_RESIDUAL=1`, set before `merlin serve`, computes it in
  log space instead. It changes the numerics, so output is not identical to the default. Only `0` and `1` are
  accepted.
- **Not measured on 3.0.2.1:** FGPC's shared-prefix reuse and the prefill runs on Swift1.5 and Swift1.5-Huihui (their
  decode and prefill rows ran with both on); sessions held at once, the residency panels and the 8 MP image panels other than the 20 GB row's and the 27B memory floor (carried over, dated, in each bench card).
- **The 3.0.2 limits still apply** (KV RAM cache, FIM tokens, control-marker replay, compile-cache guard), as does
  the list of items not re-checked on 3.0.2.

### Docs

- Every user-facing 3.0.2.1 knob is in the table above and in README Features; the model references cover the knobs
  specific to their model.
- The hardware matrix and How to serve gain the 20 GiB class rows. The Flash-Next, Swift1.5 and Swift1.5-Huihui
  references state FGPC on, the 512-token slices and the prefill runs.
- The cache-thrash page notes that its rules also run with the KV RAM cache.

## Merlin 3.0.2

2026-11-05. Built on vLLM 0.33.0 (943 upstream commits after 0.32.0), with the same seven models as 3.0.1. Figures in
the 3.0.2 bench table were measured for this release, except those marked as carried over.

### Highlights

- **vLLM 0.33.0 base.** torch 2.14.0 (NCCL 2.30.7), torchvision 0.29.0, Triton 3.8.0, FlashInfer 0.7.2.post1,
  humming-kernels 0.1.19, Transformers 5.20.2 and Pillow 13.0.0. The CUDA 13.0 toolkit and the sm_80/86/89 build are
  unchanged.
  - **Lanes:** every lane was re-run on 3.0.2: it booted, served a 262K request and has a new
    bench card. The KV host tier reached 98.3 % (Flash-Next) and 98.2 % (Qwen3.8-27B) of the trace's ceiling.
  - **Temperature-0 output:** the 27B (two cards) and Super VL match 3.0.1 within the variation 3.0.1 shows on its
    own. Flash-Next's output differs from 3.0.1's (new torch, Triton and FlashInfer), by less than 3.0.1's own
    difference between serving one request and eight at once.
  - **Flash-Next quality A/B against 3.0.1, same session:** GSM8K (full test set) scored 1,281 of 1,319 (97.1 %)
    against 1,276 and 1,275 (96.7 %). On nine 32K needle prompts 3.0.2 answered 4 and 3.0.1 answered 6. Every miss
    was a refusal (the needle is phrased as a secret passphrase), never a wrong answer.
- **FlashInfer sm_86 FP8 patches re-ported to 0.7.2.post1** with their switches and defaults:
  - the small-query verify tile is on only where the launcher sets `FLASHINFER_SM8X_FP8_SMALLQ=cta32` (the 27B);
  - the workload plan mode stays off unless `MERLIN_FLASHINFER_WORKLOAD_TILE=1` is set at install.
- **PLE table on NVMe (experimental).** `--ple-store disk` on the Flash-Next family reads the embedding table's rows
  from the checkpoint's files instead of holding the table in host RAM.
  - **On 3.0.2, against `ram`** (`--ple fp8`, 4× RTX 3090, one Gen4 NVMe):
    - free memory after startup is 47.8 GiB higher;
    - temperature-0 output is identical;
    - decode, warm first token and 262K are unchanged within run-to-run noise;
    - with the warm-up read off, the first request after a cold start waits longer for its first token (+51 % at 10K)
      until the page cache holds the table.
  - **On the 3.0.1 base:**
    - `--ple int6` frees 38.8 GiB;
    - the first uncached send decodes 9-11 % slower;
    - with `--kv-ram-cache 96g` the table never fully warms (about +10 % decode at 262K).

  Other disks are untested ⚫. `ram` stays the default. The server stops if a table file changes while it runs.
  Details: [How to serve](how-to-serve.md#the-ple-table-on-nvme-experimental).
- **`--context SIZE` (experimental)** serves past 262K with YaRN built from the checkpoint's config (`merlin serve`,
  `./setup.sh`).
  - A bare `--max-model-len` above native now adds the same recipe.
  - A size above the layout's KV pool is refused.
  - The Flash-Next reference recipes were tested on the 3.0.1 base, not yet on 3.0.2.
- **Forged control tokens are encoded as text.** A chat message, tool or document that spells a control token (for
  example `<|im_end|>`) is encoded as plain text; only the chat template emits real control tokens.
  - On by default; `--no-sanitize-control-tokens` turns it off, and a request cannot.
  - `--reject-mm-with-string-template` (off by default) refuses multimodal input with a string-format template.
  - Ornith 1.5 also treats its six FIM tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`,
    `<|repo_name|>`, `<|file_sep|>`) as control tokens.
- **`tool_choice="none"` keeps the text the model wrote** on the Qwen-family parsers (`qwen3`, `qwen3_coder`,
  `nemotron_v3` and those built on them):
  - tool-call markup comes back in the field it was written in, where before it was dropped;
  - markup inside reasoning no longer ends reasoning early.

  Details: [engine changelog](merlin/ENGINE-CHANGELOG.md#merlin-302).
- **A Ready line, a bug-report file and client settings.**
  - `merlin serve` prints `Ready: http://HOST:PORT/v1 · model · context · key` once `/health` answers
    (`MERLIN_READY_BANNER=0` turns it off).
  - `merlin doctor --report [FILE]` writes preflight, the serve plan and the log's tail to one redacted 0600 file.
  - [Connect a client](clients.md) covers opencode, Cline/Roo, Aider and Continue.
- **Compile-cache fault guard, on by default.** After a sticky CUDA fault (illegal memory access, device-side assert,
  uncorrectable ECC), the next start moves the compile-cache directories the process used to `.merlin-quarantine/`
  and recompiles. The compiled model file is also checked against a SHA-256 sidecar. Numerics are unchanged.
  `VLLM_MERLIN_COMPILE_CACHE_GUARD=0` turns it off.

### Measured on Merlin 3.0.2

4× RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W; runs of 2026-11-04 (Swift1.5, Swift1.5-Huihui and
ThinkingCap 2026-11-05). Columns as in 3.0.1:
- decode is one stream, thinking on (effort medium where the template has levels), median of 3;
- 8 agents is the total of eight simultaneous follow-ups on 16K conversations.

\* marks a figure carried over from the 3.0.1 card. GSM8K-200 thinking on was re-run on 3.0.2 for Swift1.5,
Swift1.5-Huihui and ThinkingCap; every other GSM8K-200 figure is carried over: thinking on from the 3.0.1 run,
thinking off from the 3.0.0 run (the weights are unchanged). For Flash-Next, the full test set was run on
3.0.2 (see Highlights).

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next | 4x | 924,993 | 172 / 189 / 207 | 6,782 tok/s | 654 / 664\* | 197 / 199 |
| Swift1.5 | 4x | 924,993 | 181 / 187 / 186 | 6,762 tok/s | 659 / 650\* | 199 / 197 |
| Swift1.5-Huihui | 4x | 924,993 | 180 / 189 / 193 | 6,830 tok/s | 718 / 681\* | 198 / 198 |
| Qwen3.8-27B | 1x | 271,506 | 131 / 100 / 77 | 586 tok/s | n/a (two at most) | - / 195 |
| Qwen3.8-27B | 2x | 826,689 | 211 / 184 / 146 | 1,112 tok/s | 682 / 615\* | - / 194 |
| Qwen3.8-27B | 4x | 2,012,105 | 258 / 255 / 250 | 2,111 tok/s | 940 / 854\* | 198 / 197 |
| ThinkingCap | 2x | 826,689 | 205 / 169 / 160 | 1,115 tok/s | 708 / 633\* | - / 193 |
| ThinkingCap | 4x | 2,012,105 | 238 / 235 / 223 | 2,108 tok/s | 939 / 832\* | 198 / 195 |
| Nemotron 3.5 Super VL | 4x | 1,856,367 (the shipped 1M context) | 101 / 101 / 106 | 4,217 tok/s | 432\* / 432\* | 197 / 195 |
| Ornith 1.5 | 4x | 894,618 | 145 / 122 / 111 | 2,288 tok/s | 528\* / 486\* | 196 / 190 |

Against 3.0.1:
- **Super VL:** 262K decode is 14 % faster (5 % with thinking off), and cold prefill is 8-9 % faster at every depth.
- **The other lanes** are within run-to-run spread. MTP acceptance moves decode by several percent between sends
  (Flash-Next's three 262K sends ran 184-213 tok/s), so compare step time: Flash-Next's is 1.2-1.9 % shorter than
  3.0.1's in every row.
- **A few rows read 2-4 % lower,** mostly thinking-off 4K. Swift1.5-Huihui's thinking-off 262K reads 10 % lower, from
  MTP acceptance: its step time is level, and the 3.0.1 figure is inside the spread of its three 3.0.2 sends.
- **ThinkingCap's four-card 262K reads 6 % lower** with thinking on, also from MTP acceptance (4.00 against 4.27
  tokens per step; step time 19.60 against 19.56 ms). All three 3.0.2 sends read below the 3.0.1 figure; the 259K
  ladder cell reads 8 % higher.

Every row is in the bench cards:
- [Flash-Next](../models/flash-next/benchmarks/2026-11-04/BENCH-CARD.md);
- [Swift1.5](../models/swift1.5/benchmarks/2026-11-05/BENCH-CARD.md);
- [Swift1.5-Huihui](../models/swift1.5-huihui/benchmarks/2026-11-05/BENCH-CARD.md);
- [Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-04/BENCH-CARD.md);
- [ThinkingCap](../models/thinkingcap/benchmarks/2026-11-05/BENCH-CARD.md);
- [Super VL](../models/nemotron-super-vl/benchmarks/2026-11-04/BENCH-CARD.md);
- [Ornith 1.5](../models/ornith1.5/benchmarks/2026-11-04/BENCH-CARD.md).

### Before you upgrade

- **New wheel and venv.** The wheel is `vllm-0.33.0+merlin.665f43d2cf-cp313-cp313-linux_x86_64.whl`, and `vllm --version`
  prints `0.33.0`; a 3.0.1 wheel does not run the 3.0.2 checkout. `./setup.sh` and `release/install-env.sh` build a new
  venv from `release/requirements-0.33.lock`; they do not upgrade one in place.
- **The first start of each model recompiles** (new torch and Triton cache keys).
  - `CACHE_ROOT` moves the cache on every launcher.
  - `VLLM_CACHE_ROOT` moves it only for the 27B.
- **`--mamba-cache-mode` has no effect on 0.33.** Prefix caching follows `--enable-prefix-caching`. The launchers still
  pass it, so the server logs one harmless deprecation warning. If you pass it yourself, drop it.
- **`merlin serve --instances N` stays in the foreground.** Ctrl+C, SIGTERM or a closed terminal stops every server,
  and if one exits the others stop. Scripts that used `--wait-ready S && next` must background it and poll `/health`.
- **New refusals:**
  - a checkpoint with missing quantization scales (wNa16, ModelOpt FP8);
  - a Flash-Next-family directory with neither the release scales file nor the verified marker. `config.json` is
    identical across the family. Pass `SCALES=<file>` (the refusal names it) or run `merlin pull` / `./setup.sh`;
  - `--tokens-only`, `--skip-tokenizer-init` and their abbreviations;
  - a request with `n` above `min(VLLM_MAX_N_SEQUENCES, max_num_seqs)` (HTTP 400).
- **The P2P mode alone sets the NCCL and IPC environment.** Multi-card profiles clear an inherited `NCCL_P2P_DISABLE`,
  `NCCL_P2P_LEVEL`, `NCCL_PROTO`, `NCCL_ALGO` and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`, with a note for each. Choose
  with `P2P=on|off|ipc` or `--profile`.
- **`VLLM_SERVER_DEV_MODE` is stripped** by every launcher. `MERLIN_ALLOW_SERVER_DEV_MODE=1` lets it through, with a
  warning that those routes have no API-key protection.
- **`VLLM_DRAFT_VOCAB_FILE` and `VLLM_DRAFT_VOCAB_SHA256` do nothing**: the reduced-draft-vocabulary path is removed.
- **Ornith's `serve.sh` reads peer access.**
  - `P2P=on` or `P2P=off` picks the row.
  - Otherwise the cards decide.
  - If peer access cannot be read, it stops with "set P2P=on or P2P=off".
- **Tested status follows the P2P mode.** The CMP 170HX and RTX 4090 rows (P2P-off evidence) report untested with P2P
  on.
- **The KV RAM cache "untested" note now starts above 48g** (160 GiB before). Nothing is refused.
- **Prefix-cache hashes and hit lengths are unchanged from 3.0.1** (upstream's lookahead block hashes are pinned off).

### Fixes

- **The container builds without an NVIDIA driver** (it links the toolkit's `libcuda` stub), and `MERLIN_WHEEL_URL`
  installs the release wheel instead of compiling. Found and fixed by brandon-nuttall (halt95/merlin-engine#10). They
  also qualified the 3.0.1 image on 8× RTX 3090 at `P2P=off`; the 3.0.2 image is not yet requalified.
- **A TP group of mixed cards (RTX 3090 with 3090 Ti) now boots more than once with the PCIe-IPC all-reduce.** Before,
  FlashInfer's tune cache recorded only rank 0's card name, so later boots failed. Now the ranks agree on loading it,
  and a mixed group's cache is named after its card set. Reported by
  [brandon-nuttall](https://github.com/brandon-nuttall).
- **`P2P=ipc`** (opt-in) is for Flash-Next hosts where NCCL's P2P bootstrap hangs but CUDA peer access works: NCCL
  through shared memory plus the PCIe-IPC all-reduce.
  - Refused on rows without the PCIe-IPC all-reduce, and by `merlin doctor --p2p ipc` without peer access or with
    BAR1 under 1 GiB.
  - Checked on the reference host (decode within 1 % of `P2P=on`) but not on a host with the hang, so it reports
    untested.

  Thanks to [brandon-nuttall](https://github.com/brandon-nuttall).
- **The PCIe-IPC all-reduce checks peer access across the whole TP group,** so every rank takes the same fallback.
- **Super VL with P2P on** keeps vLLM's custom all-reduce and uses `NCCL_PROTO=Simple` instead of `LL`. Output is
  identical, and a 20K prefill was about 6 % faster (torch 2.13).
- **One request can no longer stop a lane through the sampler limits.** Oversize `allowed_token_ids`, `logit_bias`,
  `bad_words` or `stop_token_ids` that bypass the HTTP frontend now fail only that request. The limits are unchanged.
- **Transient failures return HTTP 503 instead of 500:** a multimodal cache miss and, with KV connectors, a KV load
  failure under `kv_load_failure_policy=fail`.
- **Parsers:**
  - the Qwen-family parsers keep literal markers as text in non-streaming replies, as streaming already did;
  - `qwen3_coder` (also used by `nemotron_v3`, MiMo, Step3p5 and MiniCPM-V) keeps literal tags in values and
    reasoning as text, and handles truncated arguments;
  - with `validate_tool_names=True`, `qwen3_coder` drops calls to names the request did not offer.
- **Bounds and security:**
  - top-k logprob counts above the vocabulary size are rejected;
  - pending UTF-8 detokenization is bounded;
  - error-message path masking is one linear pass;
  - the structured-decisions endpoint honours `--trust-request-chat-template`.
- **Full-mode KV reuse events** no longer raise `IndexError` on a partial Mamba hit.
- **From the 0.33 merge:**
  - cuBLAS handles are created before profiling;
  - EAGLE/MTP drafts are skipped on steps where no request can use them;
  - long-context kernels use 64-bit address arithmetic, with bit-identical outputs.

### Engine

- Base: vLLM 0.33.0 (upstream commit `4ba787ec99`).
- **Pipeline-parallel decode with RecoverSSM**, with tokens unchanged by construction:
  - the commit kernels are warmed on every pipeline rank at boot, so the first commit no longer loads a Triton kernel
    mid-serve;
  - duplicate align-postprocess launches are dropped;
  - the deferred commit inputs reuse a snapshot ring.

  Not A/B-tested on their own; the bench cards include them.
- Qwen3.5-class models expose auxiliary hidden states over pipeline parallelism (EAGLE3-style drafters need this).
- Speculative verification skips random draws for all-greedy batches and skips fully masked vocabulary blocks
  (bit-identical).
- The PLE format guard accepts upstream's NVFP4 method with `uint8` tables.

### New flags and environment variables

| Name | Where | Default | What it does |
|---|---|---|---|
| `--ple-store ram\|disk` | `merlin serve`, `./setup.sh` | `ram` | **Experimental.** Where the Flash-Next PLE table lives. `PLE_WARM=0` turns the warm-up read off. |
| `--context SIZE` | `merlin serve`, `./setup.sh` | native | **Experimental.** Serves SIZE tokens (`512k` = 524,288), with YaRN past 262K. |
| `merlin doctor --report [FILE] [--log FILE] [--lines N]` | `merlin doctor` | off | One redacted bug-report file. |
| `MERLIN_READY_BANNER=0` | launcher | on | Turns the Ready line off. |
| `--sanitize-control-tokens` / `--no-sanitize-control-tokens` | vLLM | on | Encodes control tokens in chat content as text. |
| `--reject-mm-with-string-template` | vLLM | off | Refuses multimodal input with a string-format template. |
| `VLLM_MERLIN_EXTRA_CONTROL_TOKENS` | engine | empty | Extra tokens to treat as control tokens (Ornith sets its FIM tokens). |
| `VLLM_MERLIN_COMPILE_CACHE_GUARD` | engine | on | `0` turns the compile-cache fault guard off. |
| `P2P=ipc` | launcher | off | See Fixes. |
| `MERLIN_ALLOW_SERVER_DEV_MODE=1` | launcher | off | Lets `VLLM_SERVER_DEV_MODE` through. |
| `MERLIN_WHEEL_URL`, `MERLIN_PROFILE` | `container/` | empty | Install the release wheel; start `merlin serve --profile <id>` (Flash-Next family). |
| `MERLIN_LIBCUDA` | `release/install-env.sh` | empty | The toolkit's `stubs/libcuda.so`, for a source build without a driver. |
| `VLLM_MERLIN_DECODE_FIRST=1` | engine | off | **Experimental, not proven token-identical.** Keeps running sessions decoding during a long cold prefill; pair with `-- --prefill-schedule-interval N`. Measured on Flash-Next during a cold 64K-128K prefill: sessions kept 28 % of their decode speed (10 % by default), but the cold prompt's first token took twice as long. Reported by Sudzpel (halt95/merlin-engine#6). |
| `VLLM_MERLIN_PP_DECODE_BALANCE=1` | engine | off | **Experimental.** Caps decode rows per step at `ceil(decoding / pp_size)`; tokens unchanged. On the 3.0.1 base, Flash-Next TP2×PP2 measured every row within 3 % of off (no gain, no loss). |
| `VLLM_RECOVERSSM_PP_SPLIT_RECV=1` | engine | off | Rank 0 commits the deferred RecoverSSM step once the token counts arrive. Not measured. |
| `VLLM_MERLIN_QSA_SCRATCH=off\|reserve\|workspace` | engine | `off` | Lets KV sizing see the QSA indexer's scratch. Same kernels; only KV capacity moves (not measured). |
| `VLLM_SPEC_STABLE_RESIDUAL=1` | engine | off | Underflow-safe rejection-sampler residual. Changes numerics. |
| `VLLM_FLA_PACKED_DECODE_VERIFY_PARITY=1` | engine | off | Packed GDN decode normalises like verify. No shipped lane uses the packed kernel. |
| `MERLIN_MARLIN_DENSE_CFG` | engine | unset | Exact-shape W8A16 Marlin tile rules. No entry ships. |
| `MERLIN_KV_HOST_TIER__MAX_PENDING_STORE_BLOCKS`, `MERLIN_KV_HOST_TIER__FALLBACK_BOUNDARY` | engine | off | Cap on held write-back references; keep a successor's predecessor as its restore fallback. |

### Held back from 0.33

Each waits for a hit-rate or token-identity comparison:
- **Lookahead block hashes (vllm#61589)** are pinned off; they change the prefix-hash format and hit length on MTP
  lanes. `"disable_lookahead_block_hashes": false` in `--speculative-config` turns them on (unsupported).
- **DFlash and DSpark keep dropping the trailing block** (vllm#54163).
- **Small-batch split top-p (vllm#61996) stays reverted.**
- **Block verification keeps the 3.0.1 RNG keying.** `VLLM_SPEC_STEP_KEYED_RNG` opts into upstream's fix
  (vllm#61161).
- **Upstream's Ampere FP8 QSA paths** are not adopted; Merlin's own reader stays.

### Known limits

- **`tool_choice="auto"` can still return quoted tool-call markup as a real tool call (open).** Send
  `tool_choice="none"` for explain-only requests. With `response_format` and thinking on, a quoted `<tool_call>` in
  reasoning can leave the JSON in `reasoning`.
- **Only Ornith treats its FIM tokens as control tokens.** The other Qwen-family checkpoints can still have them
  forged.
- **Quoted control markers can re-encode differently on replay** (transformers#49891) when a user message spells a
  marker next to punctuation or code.
- **The compile-cache guard cannot see a fault that kills the process before Python does.**
- **The KV RAM cache limits from 3.0.1 still apply.**
- **Not re-checked on 3.0.2:** the NCCL/P2P sweep; the Triton autotune requalification; the 27B 8 MP image
  check.

### Docs

- Install lines, badges and lock references name 0.33.0, `release/requirements-0.33.lock` and FlashInfer 0.7.2.post1.
- Every model card gains a 3.0.2 bench-card section.

## Merlin 3.0.1

2026-11-02. Built on vLLM 0.32.0, with the same seven models as 3.0.0. Every figure in the 3.0.1 bench tables was re-measured for this release
with the GPU unlock and P2P on release candidates just before the final commit (see below), except rows marked
(3.0.0 run). The DFlash2 and KV RAM cache figures were measured separately on earlier builds.

### Highlights

- **KV RAM cache (experimental).** `--kv-ram-cache 24g` keeps conversation prefixes that the GPU evicts in host RAM,
  and restores them instead of recomputing them. A restore gives exactly the same output as a GPU cache hit. It works on
  the Flash-Next family (MTP drafter only) and Qwen3.8-27B. It's off by default; Ornith 1.5 accepts it but is untested.
  On Flash-Next a 262K restore takes 1.2–1.4 s instead of about 45 s to recompute. Known limits: a 262K session
  evicted during heavy prefill may be recomputed instead of restored, and host memory grows in steps over the first
  262K requests while the request history fills (bounded: about 70 MiB on Flash-Next, 280 MiB on the 27B; in our soaks
  0.18–0.38 % per hour of resident-memory growth while it fills, above the 0.1 % per hour soak limit).
- **More sessions with `--swarm` (experimental).** Raises the session cap on four RTX 3090s, with or without
  P2P: from 8 to 24 on Flash-Next, Swift1.5 and Swift1.5-Huihui, and from 24 to 32 on Qwen3.8-27B and ThinkingCap. It
  turns on a 48 GB KV RAM cache with it. Tested on Flash-Next and the 27B with P2P; Swift1.5 and Swift1.5-Huihui, and
  every model without P2P, use the same cap untested (⚫, expected to work), and the server prints a warning on the
  no-P2P rows. Default caps are unchanged, and other models and layouts don't have a swarm
  cap yet.
- **DFlash2 drafter for the Flash-Next family.** `--spec dflash2` on 4× 3090 with P2P. Decode is 5–18 % faster than MTP,
  and the server holds three 262K sessions instead of six 131K ones. MTP stays the default.
- **GPU unlock support.** [Zagitta's gpu-unlock](https://github.com/Zagitta/gpu-unlock) is documented as an optional
  host upgrade, like P2P (proposed, with before/after measurements, by Sudzpel). In a stock-versus-unlocked A/B on the reference host at 220 W it measured about 16 % faster
  prefill on Flash-Next and 29 % on Qwen3.8-27B (two cards, 262K), with decode unchanged and identical output. From 3.0.1 on, our bench numbers are
  measured with it.
- **Longer context with YaRN.** Flash-Next serves beyond 262K with a YaRN override, tested on the reference host at
  393K and 524K (MTP and DFlash2) and 917K (MTP). The 1M recipe and YaRN on the 27B are untested. Asking for more than
  262K without YaRN is now refused at startup. This applies to every model: a context longer than the model's RoPE
  table without YaRN is refused, and `VLLM_ALLOW_LONG_MAX_MODEL_LEN` no longer overrides it.
- **INT6 embedding table.** `--ple int6` loads a smaller embedding table for Flash-Next and Swift1.5, saving about
  9 GiB of host RAM per server with quality checks passed. It is one shared pack laid over your FP8 checkpoint, which
  stays untouched; `merlin verify <dir> --ple int6` checks it, and `MERLIN_PLE_PACK_DIR` points at a local copy. FP8
  stays the default. Swift1.5-Huihui is refused everywhere (setup, pull, serve and its launcher) and serves with
  `--ple fp8`, because its MTP acceptance drops too far.

### Measured on Merlin 3.0.1

4x RTX 3090 without NVLink, P2P on, Zagitta's gpu-unlock, 220 W, 2026-11-02 bench cards (runs measured 2026-11-01 to 2026-11-02 UTC). Decode: one stream, thinking on (reasoning effort medium where the checkpoint has one; the Nemotron 3.5 Super VL and Ornith 1.5 templates have no effort levels), median of 3, at a 4K / 131K prompt and the full 262,144-token context (261,888 prompt + 128 out). 8 agents: total throughput of eight simultaneous follow-ups on 16K conversations; with thinking on a 512-token follow-up may end inside the reasoning, and reasoning tokens count. GSM8K-200 is correct answers out of 200: thinking on at effort medium on this release (on four cards), thinking off from the 3.0.0 runs, as the weights are unchanged.

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents, tok/s, thinking on / off | GSM8K-200, thinking on / off |
|---|---|---|---|---|---|---|
| Flash-Next | 4x | 924,993 | 178 / 183 / 197 | 6,803 tok/s | 677 / 664 | 197 / 199 |
| Swift1.5 | 4x | 924,993 | 176 / 186 / 186 | 6,783 tok/s | 673 / 650 | 199 / 197 |
| Swift1.5-Huihui | 4x | 924,993 | 178 / 189 / 196 | 6,866 tok/s | 703 / 681 | 198 / 198 |
| Qwen3.8-27B | 1x | 271,506 | 125 / 95 / 71 | 579 tok/s | n/a (two at most) | - / 195 |
| Qwen3.8-27B | 2x | 826,689 | 210 / 182 / 143 | 1,115 tok/s | 705 / 615 | - / 194 |
| Qwen3.8-27B | 4x | 2,012,105 | 251 / 243 / 238 | 2,101 tok/s | 954 / 854 | 198 / 197 |
| ThinkingCap | 2x | 826,689 | 203 / 162 / 160 | 1,114 tok/s | 697 / 633 | - / 193 |
| ThinkingCap | 4x | 2,012,105 | 244 / 230 / 238 | 2,099 tok/s | 952 / 832 | 198 / 195 |
| Nemotron 3.5 Super VL | 4x | 1,856,367 (the shipped 1M context) | 103 / 102 / 93 | 3,916 tok/s | 432 / 432 | 197 / 195 |
| Ornith 1.5 | 4x | 894,618 | 146 / 115 / 105 | 2,261 tok/s | 528 / 486 | 196 / 190 |

Against 3.0.0, 262K prefill is 16-17 % faster on the Flash-Next family and 27-33 % faster on the other checkpoints, and four-card 27B time to first token at 262K is 125 s (164 s in 3.0.0). The four-card 27B and ThinkingCap agent totals and the two-card 27B thinking-on total come from overlap windows under 5 s; read them loosely, as on the 3.0.0 cards. Each checkpoint's card is under `models/<checkpoint>/benchmarks/2026-11-02/`, for example [Flash-Next](../models/flash-next/benchmarks/2026-11-02/BENCH-CARD.md) and [Qwen3.8-27B](../models/qwen38-27b/benchmarks/2026-11-02/BENCH-CARD.md).

### Hardware

- **More hardware.** On 4× RTX 3090 without P2P, Ornith 1.5 and Flash-Next's PP4 layout are tested; Flash-Next's
  default layout and the 27B on four cards were boot-checked with `--swarm` (2026-11-01). 4× RTX 4090 and 2× / 4× CMP 170HX
  64 GB are 🟡 testing: full model runs passed on cloud-hosted cards and kernel checks are still to run, so `merlin serve`
  picks them like untested profiles. Tuning continues on all three. 4× CMP 170HX (`fn-sm80-64g-x4-pp4`) was 🟢 tested
  in 3.0.0 and moves to 🟡 testing until its kernel checks run; on a host that matches both a tested and a testing
  profile, `merlin serve` picks the tested one.
- **Flash-Next PP4 on four cards without P2P**, by name: `--profile fn-quad-3090-nop2p-pp4`. Prefill is 47–53 % faster
  and decode 14 % slower than the default TP2×PP2, which stays the default. Its KV pin is now 4.15 GiB per card: a
  906,581-token pool that holds three 262K sessions (one before), tested on the reference host on 2026-10-19.
  `--profile fn-quad-24g-sm89-pp4` is the same layout on 4× RTX 4090 without P2P, by name only and untested ⚫.
- **4× RTX 4090 prefills in 2,048-token chunks** (`fn-quad-24g-sm89`; 1,024 before). On cloud-hosted 4090s this cut the
  131K time to first token from 24.4–27.0 s to 22.5 s (about 8–20 % faster prefill), with decode unchanged. The 8 MP
  image and 262K checks ran at 1,024 and have not run at 2,048 yet; `CHUNK=1024` goes back to the checked chunk.
- **FP8 KV cache for the Flash-Next family on sm_80 (A100, A30, CMP 170HX)** as an opt-in with `KV_DTYPE=fp8`. BF16
  stays its default there (the 27B already uses FP8). Untested.
- **Host RAM for Flash-Next on four cards:** 72 GiB is the tested minimum (64 GiB ran out of memory). Startup is
  refused below about 69 GiB, plus the KV RAM cache size when it's on; with `--ple int6` the floor is about 60 GiB
  (estimated, not boot-tested). 96 GB is the reference configuration; a host with at least 92 GiB MemTotal counts as
  that class. `MERLIN_SKIP_RAM_CHECK=1` keeps a RAM refusal as a warning only (see SETUP).
- **Qwen3.8-27B on WSL2 and on cards without P2P.** On WSL2, Merlin turns on pinned memory, which the 27B,
  Nemotron 3.5 Super VL and Ornith 1.5 need for their host embedding tables, and checks for WSL before starting;
  on Flash-Next it prints how to turn it on instead. `P2P=off` selects the no-P2P layouts.

### Before you upgrade

- **The engine wheel is unchanged from 3.0.0** (no compiled-code changes in 3.0.1; the native sources changed only in comments and message text), so the wheel file is still
  `vllm-0.32.0+merlin.7002f56537-…`, and after the install `vllm --version` prints `0.32.0`; the 3.0.1 Python comes from the checkout.

- **Request limits.** `allowed_token_ids` or `logit_bias` over 1,024 entries, `bad_words` over 128 sequences or 1,024
  tokens in total, or `stop_token_ids` over 128 with `min_tokens` set now return HTTP 400 instead of crashing the
  engine. With speculative decoding, `logit_bias` is refused. Generative
  scoring takes up to 32 items and 32,768 tokens (`MERLIN_GENERATIVE_SCORING_MAX_ITEMS`, `..._MAX_TOKENS`).
- **The API key now also covers** `/tokenize`, `/detokenize` and `/generative_scoring`. `/tokenize` refuses media
  parts.
- **An explicit `VLLM_HOST_EMBED_TABLE=1` is refused at boot** on a model whose embeddings are tied or quantized
  (it used to be skipped silently): unset it or set it to `0`. Unset, the default, is unchanged, and Flash-Next is
  unaffected.
- **The first Qwen3.8-27B start recompiles**, because its compile cache moved into the install (`.vllm-cache`);
  `VLLM_CACHE_ROOT` overrides it.
- **`REDUCED_DRAFT_VOCAB` and `DRAFT_VOCAB` now stop the launch** when set (an empty value, or `REDUCED_DRAFT_VOCAB=0`,
  is accepted). Reduced draft vocabulary is gone (see Removed); unset them.
- **Thinking-off decode at 262K measured lower on five layouts:** Swift1.5 −11.4 %, Qwen3.8-27B on four cards −5.7 %
  and on two cards −2.1 %, Nemotron 3.5 Super VL −1.2 % and Flash-Next −0.4 % against 3.0.0. Step time is the same or better; the gap is MTP draft acceptance over a 128-token sample, and
  the 259K thinking-off cells read +0.8 % and +5.4 % (see the bench cards).

### Fixes

- **`/health` watches for stalls**, on by default with stats logging, one API server and one engine: it returns 503
  when the engine completes no step for 120 s with requests in flight (`VLLM_MERLIN_HEALTH_STALL_S`; `0` turns it off). A failed request logs its cause at ERROR
  and returns a clear error message.
- A single request too large for the KV cache now fails with "KV cache capacity exhausted" instead of hanging.
- The RAM check now respects memory limits set on a container or a systemd service, so a server that can't fit is
  refused before the weights load instead of being killed partway through.
- A worker that dies during startup now fails the boot instead of hanging it.
- Structured output (JSON mode, strict tools, grammars) with speculative decoding is now exact. Draft acceptance
  for these requests reads lower as a result; this is expected.
- Flash-Next refuses to run FP8 KV with a recipe's automatic scales when the checkpoint's config doesn't match them; scales you set yourself with `SCALES` aren't checked. Calibrate a new checkpoint with
  `tools/qsa_calibrate.py` and pass `SCALES=`; the calibration now records which checkpoint it was made on, and the
  launcher warns when the scales name another checkpoint or were not calibrated under the YaRN length being served.
- A Flash-Next checkpoint whose config doesn't set `mamba_ssm_dtype` now keeps its GDN state in float32, as every shipped
  checkpoint does, instead of falling back to bf16.
- Bad sampler settings are rejected with an error before the request is admitted.
- An FP8 embedding table with broken scales (NaN, infinite or zero) and incompletely packed shards are refused at
  load.
- `--skip-tokenizer-init` is refused, and Merlin notes when extra arguments override a recipe value.
- `include_reasoning=false` no longer marks reasoning finished before generation starts; structured output now waits
  for the reasoning to end.
- With thinking off, a non-streaming response now puts the whole output in `content` and reports zero reasoning
  tokens, as streaming does.
- `/v1/messages`: a `tool_result` that carries a `tool_reference` part (deferred tool search) is now rendered as
  `Tool reference: <name>` text instead of failing with HTTP 400, so a history that contains one replays.
- **A boot hang on the Flash-Next family is fixed.** A Triton kernel loading while the host n-gram table was being
  pulled could stall both first-stage ranks. Every model now preloads its boot kernels and synchronises before a
  later kernel load. Two new warnings: `Triton kernel first compiled while serving` (a kernel compiled after startup,
  which costs that request time) and `Triton preload short for ...` (fewer kernels preloaded than expected).
- A dangling link in the vLLM cache directory is caught at startup and by `merlin doctor`, and the 27B keeps its compile
  cache inside the install.
- Sessions that stop while preempted, or on a hybrid model, resume correctly from the prefix cache.
- `n > 1` requests with a large seed no longer overflow.
- `merlin serve --instances N` starts all N servers or none. With `--wait-ready SECONDS` it waits until every
  instance answers `/health` and is still running; if one dies, even after answering while others load, or misses the
  deadline, all are stopped and the command exits 1. If it cannot confirm the stopped servers are gone, it names them
  so you can stop them by hand.
- **YaRN past 262K no longer overruns the drafter.** With a YaRN override, the MTP and DFlash2 drafters kept the
  checkpoint's 262,144-position table, so a longer prompt could stop the engine. The drafter now follows
  `--max-model-len`, and the draft `max_position_embeddings` override covers every eagle-family drafter (MTP,
  DFlash, DSpark) and the text config, so drafts follow the target's YaRN table past 262K (a port of the open upstream
  PR vllm#63432). Reported by brandon-nuttall and R-omk, whose 524K recipes we then tested.
- A failed `generate()` request is always logged at ERROR, and `EngineGenerateError` carries a default message.
- The "Unknown Merlin environment variable" warning no longer fires for setup and launcher `MERLIN_*` names, and the
  per-admission `SimpleCPU tier miss` log is DEBUG unless `MERLIN_KV_HOST_TIER_CENSUS=1`.
- A clean shutdown no longer prints an `EngineDeadError` traceback.
- An always-fatal bounds-guard hit (QSA rope position, uncommitted ring) found at shutdown now fails the rank with
  exit code 1 in every guard mode, including the default `warn`.
- The host n-gram table no longer times out on a cold kernel compile and leaves no orphaned process when killed.
- `install-env.sh --verify` and `--check` work without `uv` on the PATH. With an inherited `CUDA_HOME` that has no
  `nvcc`, it now prints the refusal and exits 1 instead of failing silently, and the FlashInfer patcher checks for GNU
  `patch` before downloading anything.
- The benchmark tool counts speculative-decoding output correctly, so its reported output rate changes.
- **Fixes from upstream vLLM:**
  - top-p no longer emits token 0 when a logit is forced very high, as at the end of a thinking budget;
  - split top-p sampling fixes;
  - hybrid models no longer write state through stale block-table rows left by an earlier, larger batch;
  - hybrid prefix hits: requests sharing an already-cached prompt boundary are no longer admitted one per step;
  - resumable and streaming sessions resume correctly;
  - cuBLAS initialises before memory profiling;
  - pipeline-parallel draft tokens are gathered in order;
  - adaptive verification no longer shrinks the KV cache by its profiling activations;
  - third-party Flash-Next quantizations with quantized q/k/v and a BF16 QSA indexer (such as AWQ builds) load.

### Engine

- The MTP drafter's attention layers now apply the FP8 KV scales. Output is unchanged.
- Qwen 4 readiness: Flash-Next family model types load from HF config, and the config check names a pipeline split
  that keeps the n-gram layers on the first card.
- YaRN M-RoPE with video pruning keeps the checkpoint's original correction window; the cos/sin cache still covers
  the longer positions. No shipped profile reaches this path.
- When freeing KV blocks by yielding one session, the scheduler now picks the smallest session that covers the
  shortfall instead of the first. Output is unchanged.
- With `VLLM_HOST_EMBED_TABLE=1`, the DFlash2 drafter's codebooks also move to host memory.

### Setup and defaults

- **`./setup.sh` recommends the 27B over ThinkingCap on 4× RTX 3090** when Flash-Next doesn't fit. At the served default (reasoning
  effort medium) ThinkingCap is not faster; its earlier lead was measured at xhigh. ThinkingCap stays available with `--model 27b-thinkingcap`.
- **SSE keep-alive every 15 s** on `/v1/chat/completions` and `/v1/completions`, so idle streams survive common proxy timeouts. `-- --sse-keep-alive-interval 0`
  turns it off.
- **Cleaner `./setup.sh` output**: titled sections, aligned options and a downloads table with a total. `NO_COLOR` is
  honoured. The existing questions, their defaults and saved answers are unchanged. Two questions are new and are
  asked only where they apply: **Drafter** on Flash-Next family rows with DFlash2 (default MTP) and **Swarm mode** on
  rows with a swarm cap (default no).
- **`merlin doctor` shows each GPU's PCIe link** and the IOMMU groups.
- **`merlin doctor` reports the GPU unlock state** per card (`on`, `off`, or `unknown` with the reason) and whether the `gpu_unlock` module is loaded,
  on the GeForce Ampere and Ada cards the module supports (tested on RTX 3090; the others untested ⚫). It reads the card's registers in a short-lived child process
  with a 5 s timeout and never writes them; it needs root on the host itself, and in a container, VM or WSL2 it says to check on the host. It runs on a live
  `merlin doctor` only, not with `--dry-run`, `--simulate` or `--inventory`.
- **DFlash2 with a KV RAM cache is refused** before anything downloads. The `--spec dflash2` hint now appears only on
  layouts that can run it.
- **First serve honours `CC` and `CXX`** (setup passes the compiler it found) and uses the venv's Ninja, so a host with
  only `gcc-14`, or with no system Ninja, no longer stops at `first serve needs ...`.
- **A complete checkpoint on a read-only mount** verifies instead of crashing while writing its verified marker.
- The suggested second-server command is quoted, so it works from a checkout path with spaces.
- **`merlin verify <dir>`** re-hashes every file of a checkpoint against its published checksums. `merlin serve`
  only checks file sizes at startup, and suggests `verify` when a checkpoint hasn't been verified.
- **The Flash-Next systemd unit restarts after the engine dies** (`Restart=always`), and stops retrying after five
  failed starts within an hour.
- **Containers:** the image tag matches the source release, and the Docker example binds to `127.0.0.1` instead of
  every interface.
- **CUDA 13.x other than 13.0** is accepted for an inherited `CUDA_HOME`; 13.0 is the tested toolkit.
- **Ornith 1.5 `SEQS=`** now sizes its CUDA-graph capture to match.
- **Launcher defaults:** `VLLM_NO_USAGE_STATS=1` on the Flash-Next family and `HF_HUB_OFFLINE=1` on the 27B and
  Ornith 1.5.
- **Launcher knobs (thanks to brandon-nuttall):** `SEQS` and `MAXLEN` now also apply on Flash-Next and the 27B (the CUDA-graph capture follows
  `SEQS`, and a value off the tested row prints a note), and Flash-Next takes `PP_PARTITION` and passes
  `MERLIN_MARLIN_MOE_CFG` through. An extra `--speculative-config` after `--` (any spelling vLLM accepts: `--speculative_config`, `-sc` or `--speculative-config.KEY`) merges with the
  recipe's; changing its method or K is refused. `SPEC=none` turns speculation off for diagnosis only; it is not a serving lane.
- **Launcher checks:** `/dev/shm` is checked for free space rather than its total size; a `MODEL` and a
  `MERLIN_CHECKPOINT_DIR` that name different folders are refused; the 27B with P2P on drops an inherited
  `NCCL_P2P_DISABLE=1`; and a server on a non-loopback `HOST` warns that the API key does not guard `/invocations`
  or the ops routes.
- **Opt-in FlashInfer workload tile for the 27B** (default off): `MERLIN_FLASHINFER_WORKLOAD_TILE=1
  release/install-env.sh` applies FlashInfer patch 0002 after 0001; see the
  [patch notes](../release/patches/flashinfer-0.7.0.post1/README.md). The Flash-Next family ignores the knob and prints a note saying so.
- **`./setup.sh` takes an install lock** (`.merlin/setup.lock`): a second run in the same checkout stops with a
  message. A successful install removes only its own rollback environment and keeps older `venv-old-*` ones. A failed
  uv size probe reports the URL and an offline/proxy hint.
- **The old `scripts/serve*.sh` shims are kept through Merlin 3.0.x** (the 3.0.0 notes said the next release would
  remove them).
- **ThinkingCap containers serve as `qwen38-27b`**, as bare metal does; `SERVED=thinkingcap` gives the old name.
- The LXC `provision.sh` no longer needs `TILE_PROOF_LOG` to reuse an environment; `install-env.sh --verify` is the
  check.
- PLE offload warns when either pull-speed control is off, including a direct `vllm serve` without the launcher's
  `VLLM_E2_PLE_FAST_IDS=1` and `VLLM_E2_PLE_EARLY_REQUEST=1`.

### Removed

- `VLLM_SPEC_DRAFT_UNDER_TARGET_TOPK`, which was off by default and measured 2–2.5 % slower. Setting it prints a note
  that it is ignored.
- The reduced MTP draft vocabulary: the `REDUCED_DRAFT_VOCAB` / `DRAFT_VOCAB` switches, the bundled table and its
  builder. It was off by default in 3.0.0. Setting either switch (other than an empty value or `REDUCED_DRAFT_VOCAB=0`) now stops the launch with a
  message; `DRAFT_HEAD` still sets the draft head's precision. A direct `vllm serve` still honours
  `VLLM_DRAFT_VOCAB_FILE`; that engine path is dormant and goes in a later release.

### Docs

- The 3.0.0 Flash-Next prefill gain (+12–13 %) is now stated against v2.5.2, which it was measured against.
- The Nemotron 3.5 Super VL host-RAM embedding table is documented. `VLLM_HOST_EMBED_TABLE=0` keeps it on the GPU.
- The 3.0.0 bench cards gained 262K rows on 2026-10-22, from the same runs.

## Merlin 3.0.0

2026-10-22. Built on vLLM 0.32.0. One engine, one install and one command for all seven checkpoints, in one repository.

### In brief

- **Seven checkpoints, one engine.** Flash-Next, Swift1.5, Swift1.5-Huihui, Qwen3.8-27B, ThinkingCap and two new ones,
  Nemotron 3.5 Super VL and Ornith 1.5 (ornith-ai's Qwen3.5-122B-A10B fine-tune), each with a recipe under `models/`.
  Every one runs at its native context (262,144 tokens; Nemotron 3.5 Super VL 1,048,576) with an FP8 KV cache, MTP
  speculative decoding and vision.
- **One command.** `./setup.sh` checks the host, recommends a checkpoint and layout for your GPUs, shows every download
  with its size, installs the prebuilt engine (about 1.5 minutes, no compiling, nothing system-wide) and starts the
  server. Answers can be changed before anything downloads. Walkthrough: [How to serve](how-to-serve.md).
- **One launcher.** `merlin serve` identifies the checkpoint and picks the largest tested layout for the cards it finds
  (`--tp`, `--profile`, `--instances N` for several servers on one host); `merlin doctor` checks the host first.
- **Prebuilt engine for Ampere and Ada.** One wheel for sm_80, sm_86 and sm_89, checked against its published sha256.
- **More hardware support.** Profiles for RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB), RTX A6000 / A40,
  A100, A30 and CMP 170HX; see [Which checkpoint runs where](how-to-serve.md#which-checkpoint-runs-where).
- **Faster prefill.** Flash-Next cold prefill 5,020 → 5,660 tok/s at 10K and 5,417 → 6,051 tok/s at 100K against
  v2.5.2 (+12–13 %); Qwen3.8-27B on four cards 2,483 → 3,009 tok/s at 10K (+21 %) against v1.3.0.
- **RecoverSSM on every checkpoint**, including 27B on four cards (pool 1,974,072 → 2,012,105 tokens) and both new
  ones, and **27B on one RTX 3090**: a 271,506-token pool, one 262K or two 131K sessions, images up to 4 MP.

### Measured on Merlin 3.0.0

4× RTX 3090 without NVLink, P2P on, 220 W, 2026-10-20 bench cards (27B one- and two-card agents/GSM8K, ThinkingCap two-card and Nemotron agents/GSM8K are from a
2026-10-17 pre-release build). Decode: one stream, thinking on, median of 3, at a
4K / 131K prompt and the full 262,144-token context (261,888 prompt + 128 out).

| Checkpoint | GPUs | KV pool, tokens | Decode, tok/s, 4K / 131K / 262K | 262K prefill | 8 agents | GSM8K-200 |
|---|---|---|---|---|---|---|
| Flash-Next | 4× | 924,993 | 168 / 170 / 184 | 5,838 tok/s | 638 tok/s | 199 |
| Swift1.5 | 4× | 924,993 | 177 / 180 / 177 | 5,834 tok/s | 643 tok/s | 197 |
| Swift1.5-Huihui | 4× | 924,993 | 187 / 188 / 189 | 5,880 tok/s | 660 | 198 |
| Qwen3.8-27B | 1× | 271,506 | 124 / 95 / 73 | 456 tok/s | n/a (two at most) | 195 |
| Qwen3.8-27B | 2× | 826,689 | 207 / 173 / 142 | 858 tok/s | 517 tok/s | 194 |
| Qwen3.8-27B | 4× | 2,012,105 | 250 / 235 / 215 | 1,596 tok/s | 752 tok/s | 197 |
| ThinkingCap | 2× | 826,689 | 198 / 182 / 165 | 860 tok/s | 502 tok/s | 193 |
| ThinkingCap | 4× | 2,012,105 | 253 / 255 / 248 | 1,594 tok/s | 733 tok/s | 195 |
| Nemotron 3.5 Super VL | 4× | 1,392,275 at 262K (1,856,367 at the shipped 1M context) | 97 / 101 / 91 | 2,951 tok/s | 429 tok/s | 195 |
| Ornith 1.5 | 4× | 894,618 | 144 / 106 / 100 | 1,704 tok/s | 468 | 190 |

Ornith 1.5 was measured on 2026-10-21 with its engine changes (uniform MoE verify on) before they merged into 3.0.0;
its bench card is [models/ornith1.5/benchmarks/2026-10-21](../models/ornith1.5/benchmarks/2026-10-21/BENCH-CARD.md).

### New checkpoints

- **Nemotron 3.5 Super VL** (NVIDIA, 120B-A12B MoE): W4A16 routed experts with an INT4 MTP head, tensor parallel 2 ×
  pipeline parallel 2 with expert parallel on four cards, RecoverSSM on its Mamba-2 path. It serves its native
  1,048,576-token context by default (pool 1,856,367 tokens): a 1,048,052-token needle came back exact and, re-sent, was
  served from the resident cache (1,047,936 tokens cached, 3 s to first token). At a 262K context the pool held 5 × 262K
  and 9 × 131K sessions at once with 0 preemptions; needles 23/23 at 262K and 10/10 at 131K (2026-10-17 pre-release build). One image per prompt,
  downscaled to the model's native ~3.4 MP.
- **Ornith 1.5** (ornith-ai's Qwen3.5-122B-A10B fine-tune, 122B MoE): INT4 routed experts, INT4 MTP experts, INT8
  GDN projections and calibrated FP8 KV scales; tensor parallel 2 × pipeline parallel 2 with expert parallel,
  RecoverSSM, MTP K=3 with uniform MoE verify (the verify batch through one MoE graph). The INT4 MTP experts accept as the BF16 ones do (mean acceptance length 2.943 against 2.947). Pool
  894,618 tokens (3 × 262K); three ~262K prompts sent at once all came back with their needles; a recorded tool-use
  session replayed with 52/52 tool calls valid. Tested with P2P on the reference host and without P2P on a second
  4× RTX 3090 host (pool 977,838 tokens, GSM8K-200 193/200). Checkpoint about 66.4 GiB.
- **Qwen3.8-27B on one card.** The single-card layout returns, tested on an RTX 3090 with Qwen3.8-27B: pool 271,506
  tokens, decode 124 / 95 / 73 tok/s at 4K / 131K / 262K, 262K prefill 456 tok/s and, on the 2026-10-17 pre-release build, GSM8K-200 195/200 and needles 5/5 at
  131K and 12/12 at 262K. ThinkingCap serves on the same profile but was not run on one card.

### One install, one command

- **`./setup.sh`** fetches uv and a Python 3.13 into the checkout when the host has none, then checks the host and
  names the exact fix for each failure: driver R580 or later, GPUs and P2P, GCC/G++ 13 or newer, disk against the
  plan, host RAM for the model. It asks one question at a time, with Enter taking the answer it recommends for the
  hardware it found: model, layout, models folder, local or network access (with a generated API key) and port.
- **Every download shown before it starts:** the engine wheel, the 222 locked packages and the checkpoint, each with
  its size, then one confirmation. The checkpoint download resumes, is pinned to one published revision and is checked
  against the release checksums; a marker skips the full re-hash while the files are unchanged (`--rehash` forces it).
  Setup ends with `merlin doctor`, the endpoint, a curl test and the server in the foreground.
- **Re-runs and updates.** `./setup.sh` again starts with the saved answers and asks nothing; `git pull && ./setup.sh`
  reinstalls only when the wheel, lock or commit moved and never downloads the model again; a failed reinstall
  restores the previous environment. `--reconfigure`, `--yes --model M` and `--dry-run` (fetches and writes nothing)
  are there for scripted installs, and `merlin pull <model>` downloads and verifies a checkpoint without a GPU.
- **Careful with your host.** Answers are kept per model, and a folder that holds another checkpoint is refused. The
  API key is stored owner-only, shown once and never printed in commands. uv's archive is checked against its
  published sha256. Disk is budgeted per filesystem.
- **`merlin serve <checkpoint>`** identifies the checkpoint, picks the largest tested layout that fits the idle cards
  and starts the server through one launcher. Every layout is a profile; every knob (`PORT`, `HOST`, `KV_BYTES`,
  `MAX_PIXELS`, `CHUNK`, `BLOCK_SIZE`, `LPT`, ...) is an environment variable. `--dry-run` prints the resolved command.
- **Card selection follows the board.** Cards are ordered by tested model, NUMA node, host bridge and PCIe switch, read
  from `nvidia-smi topo -m`. `--instances N` cuts separate card sets over that order, each with its own ports and
  compile cache, and fills mixed hosts class by class. A layout qualified with P2P is never picked without it; the
  one-card layout and `--tp 1 --instances N` are named instead. On 4× CMP 170HX, eight agents at 16K measured 873 tok/s
  on `--tp 1 --instances 4`, 523 on two TP=2 servers and 371 on one TP=4 server.
- **`merlin doctor`** checks GPUs, driver, host RAM and shared memory, honours `GPUS` / `CUDA_VISIBLE_DEVICES`, and
  reads available memory within a container's limit. `merlin list` shows every profile with its hardware.
- **Hardware profiles** beyond the reference host: RTX 3090 without P2P, eight RTX 3090s, RTX 4090 (24 and 48 GB),
  RTX A6000 / A40, A100, A30 and CMP 170HX. 4× CMP 170HX (64 GB) is tested; the others ship untested and expected to
  work. A card below its VRAM class gets a smaller pool instead of a refusal.
- **Prebuilt wheel.** One wheel for sm_80, sm_86 and sm_89 installs without compiling. The installer accepts only the
  wheel named in the published sha256, built from the checkout's native sources; a source install builds the same three
  architectures (about 1.5 hours on 8 cores). Requirements are locked to 222 exact pins with SHA-256 hashes, with the
  CUDA 13.0 toolkit wheels. A fresh install (2026-10-21) and a from-source build (2026-10-22) passed on the reference
  host.

### Engine

- **Rebased onto vLLM 0.32.0** (an upstream release candidate, the five upstream commits of the next release candidate, then the final release's one commit,
  [a macOS-only socket-family fix](https://github.com/vllm-project/vllm/pull/62177)). Eleven earlier backports now come
  from upstream, including prefix-cache and MTP fixes, GDN metadata reuse and the 1× M-RoPE cache. The rebased engine
  was temperature-0 token- and logprob-identical to the reference build on four configurations.
- **FP8 KV cache on Ampere.** Merlin's sm_86 FP8 attention reader for Flash-Next is kept, and 27B gets the FlashInfer
  0.7.0.post1 sm_86 FP8 small-query verify tile, which restores four-card step-time parity with vLLM 0.30: inter-token
  latency at 4K / 32K / 131K 19.99 / 21.41 / 25.27 ms on two cards and 15.83 / 16.62 / 18.37 ms on four.
- **Tables in host RAM.** Flash-Next's n-gram embedding table (about 47.7 GiB) lives in system memory, one copy shared
  by every GPU. The token embedding tables of the main model and its MTP drafter are read from pinned host memory:
  frees VRAM, no speed loss (about 1.2 GiB per card on 27B at TP=2). It is now on by default for Qwen3.5 MoE too, the
  Ornith 1.5 support (`VLLM_HOST_EMBED_TABLE=0` keeps the table on the GPU).
- **RecoverSSM on every checkpoint.** [RecoverSSM](https://github.com/vllm-project/vllm/pull/58863) recovers the GDN and
  Mamba-2 state for MTP speculative decoding, on by default for all seven checkpoints.
- **Cache retention and thrash protection.** Follow-ups resume from cache on a 64-token grid. Cache-thrash fixes are on
  by default for hybrid models: cost-aware admission and session-coherent yields. Under 8 × 131K and 12 × 80K
  oversubscription, follow-up turns reused 74.7 % and 82.1 % of their prompt from cache. A cold request waits behind
  cached follow-ups only when admitting it would evict blocks they still need.
- **PCIe all-reduce.** Flash-Next on P2P hosts uses FlashInfer's PCIe all-reduce for decode and NCCL `Simple` for
  prefill, in place of vLLM's custom all-reduce with `NCCL_PROTO=LL`.
- **Marlin W8A16** replaces AllSpark for the Flash-Next INT8 DeltaNet projections.
- **Bounds guards with idle P8.** The runtime bounds guards stay on, and idle cards drop to P8: 110 W idle on four RTX
  3090s, against 446 W when the guard reader kept them awake.
- **Leaner memory use.** The prompt-logprob GPU workspace is bounded by a 256-row logits chunk (was 1,024), with the
  Flash-Next pool unchanged at 924,993 tokens. 27B shared-prefix checkpoints are evict-first, one per prefix, with no
  extra blocks held at rest. Nemotron's image preprocessor no longer opens a CUDA context on GPU 0, returning about
  260 MiB to the first card.
- **New fine-tunes run as is.** A Flash-Next-family checkpoint that ships its own FP8 KV scales is served with them
  (`SCALES=` still overrides), with no repository update.
- **Upstream protocol change.** The upstream [/v1/messages protocol change](https://github.com/vllm-project/vllm/pull/61427)
  is merged with Merlin's own edits to that endpoint.
- `use_local_argmax_reduction` works with the Flash-Next MTP drafter, contributed by
  [brandon-nuttall](https://github.com/brandon-nuttall). It is off in every shipped profile, since it needs greedy
  drafting: it gave +10–20 % eight-way aggregate decode on the contributor's eight-card rig and no gain on the
  reference layout (459 tok/s against 483 for the shipped profile).

### Per-model recipes

- **Flash-Next, Swift1.5 and Swift1.5-Huihui.** Every MTP drafter scores the full vocabulary with a BF16 draft head;
  `DRAFT_HEAD=int8` is available (6.5–7.2 % lower ms per output token at the same MTP acceptance) and planned as the
  default in a later release (3.0.1 keeps `bf16`), and no reduced draft-vocabulary table ships (`REDUCED_DRAFT_VOCAB=1` is unavailable). The Swift1.5 and Huihui launchers are
  generated from the checkpoint profile with their own KV scales, both tested on the reference host. The pool holds
  3 × 262K and 6 × 131K sessions at once, 0 preemptions, and an 8 MP image fits at the full pin. Flash-Next decode
  with thinking on is 1.4–5.8 % faster than v2.5.2 at every depth (187.4 against 177.2 tok/s at 259,587 tokens).
- **Qwen3.8-27B and ThinkingCap.** RecoverSSM on every layout, MTP K=4 with an INT4 MTP head, shared-prefix checkpoints
  on, and the 0.7.0.post1 verify tile. Four cards: pool 2,012,105 tokens (7.68 × 262K), cold prefill +15 % at 100K and
  time to first token at 255K 179.3 → 161.6 s against v1.3.0, step time unchanged. Two cards take 8 MP images (806 MiB
  worst-case free per card over a 50-image sweep, above the 524 MiB floor).
- **27B on one card at 4 MP.** The single-card layout caps images at 4 MP (4,194,304 pixels) and keeps the 262K KV
  pin, with three output-identical memory settings on by default for it alone: a 96 MiB FlashInfer workspace, the MTP
  drafter sharing the target's workspace, and the vision rotary applied in place. On an RTX 3090, 4 MP passed at one and
  two sequences and after a 262K needle with 558–563 MiB minimum free (bar 524 MiB); without the settings the card ran
  out of memory at 3 MP.
- **Nemotron 3.5 Super VL at its native 1M.** The 1,048,576-token context is the default, with RecoverSSM and MTP K=3
  at the same KV pin as the 262K pool; one full 1M session fits on four RTX 3090s, or five 262K sessions.
- **Ornith 1.5.** Two profiles, with and without P2P, both tested: `merlin serve ornith1.5 <dir>`. Uniform MoE verify is
  on (`UV=0` turns it off) and was bitwise identical to verify off at one session.

### Defaults

- **Reasoning effort `medium` on every Qwen3.8 checkpoint** (Flash-Next was `low`; Swift1.5, Swift1.5-Huihui and
  ThinkingCap were `xhigh`). A request still picks its own effort through `chat_template_kwargs`. At `medium` the MTP
  drafter lands more tokens per step than at `xhigh` and GSM8K-200 scored the same or better: Flash-Next prose 2.66
  against 2.29 and code 3.06 against 2.35; ThinkingCap prose 3.27 against 2.78 and code 3.55 against 2.65. Nemotron's
  template has no `reasoning_effort` (its own switch is `medium_effort`); Ornith 1.5's has only `enable_thinking`.
- **One chat template.** Every Flash-Next-family and Qwen3.8-27B-family recipe serves froggeric's fixed Qwen3.8 template (v22.5), which accepts system
  messages after the first turn, so coding agents keep their prefix cache across turns. `CHAT_TEMPLATE=<path>` serves
  another, `CHAT_TEMPLATE=` the checkpoint's own.
- **8 MP image cap** on every checkpoint (was 4 MP), except Nemotron (its native ~3.4 MP) and one-card 27B rows (4 MP).
- **Full-vocabulary MTP drafter** with a BF16 draft head on the Flash-Next family (`DRAFT_HEAD=bf16|int8|int4`).
  Nemotron and the 27B family use INT4 MTP heads, Ornith 1.5 INT4 MTP experts.

### Docs and repository

- **One repository.** The Merlin vLLM fork sits at the root and each checkpoint has a recipe under `models/`, every one
  at version 3.0.0; checkpoints stay on Hugging Face. One installer replaces the separate engine bundles, per-package
  patch series and compiled-ops tarball of earlier releases.
- **[How to serve](how-to-serve.md)** walks through setup end to end: the questions, talking to the server, serving on
  your network, choosing GPUs, switching models and fixing problems.
- **Bench cards** for every checkpoint, dated 2026-10-20 (Ornith 1.5: 2026-10-21): decode ladder with thinking on and
  off, exact 262K, cold prefill, sessions held at once, images, agents and GSM8K-200, with the previous release beside
  it where one exists. Earlier cards keep their original dates and toolchains.
- The README links the upstream vLLM README and documentation, with a NOTICE and a security policy.

### Fixes

- **RecoverSSM in align-mode prefix caching.** With prefix caching, RecoverSSM and MTP, a window whose accepted tokens
  crossed a block boundary could store a wrong block-boundary state (the final state was always correct), and a later
  cache hit on that block could diverge from a cold run. The boundary and final states are now stored in one pass. It
  affects earlier 27B two-card and Flash-Next releases served with prefix caching: at block size 4,096 and K=3,
  roughly 1 window in 1,365 could write such a state.
- **Engine fixes carried into 3.0.0:** the padded guard sentinel, prefix-cache keys for DCP and EAGLE, the RecoverSSM
  pipeline broadcast, the host n-gram table's boot, format validation and event lifetime, prefill logits lifetime and a
  FlashInfer staging fence.
- A prefix-cache reset that cannot complete keeps the retained follow-up prefixes.
- Sequence parallelism warns instead of refusing to start; it stays off by default.

### Upgrading

Make a fresh checkout and run `./setup.sh`: a models folder that already holds a checkpoint is checked against the
release checksums and reused rather than downloaded again. The old serve scripts (`scripts/serve-v2.5.sh` and
others) forward to the new launcher with a notice and are removed in the next release. Merlin 3.0.0 does not carry the
reduced-vocabulary INT4 draft head of an earlier, unpublished Flash-Next build.
Upgrading is recommended for anyone serving with prefix caching, for the RecoverSSM fix above. A clone of
qwen38-flash-next-3090s can point at the new name with
`git remote set-url origin https://github.com/halt95/merlin-engine.git`.

## v2.5.3

2026-10-12. Engine unchanged: Merlin engine v2.5.2.

- **New: Swift1.5-Huihui** (Flash-Next tag `v2.5.3`), Swift1.5 with huihui-ai's abliteration on its 96 dense output
  projections. It does not refuse: 29 of 278 StrongREJECT prompts refused (10.4 %). GSM8K 96.29 % against Swift1.5's
  95.98 % in the same run.
- **New: Qwen3.8-27B v1.3.0** on Merlin engine v2.5.2 plus 27B patches, two and four cards. Single-stream thinking
  decode at 4K / 32K / 128K / 255K: 206 / 198 / 172 / 144 tok/s on two cards, 258 / 249 / 238 / 226 on four.
  RecoverSSM on two cards (pool 826,689, three 262K sessions resident), pool 1,974,072 on four (seven 262K sessions).
  FlashInfer 32-row verify tile, MTP K=4, cached-token counts in every response, a per-image pixel cap.
- **New: ThinkingCap**, BottleCap AI's reasoning fine-tune of Qwen3.8-27B, served as a profile of 27B v1.3.0.
- **Qwen3.8-27B v1.3.1** (2026-10-13), documentation only: the glibc floor is 2.34 (Ubuntu 22.04, Debian 12, RHEL 9).

## v2.5.2

2026-10-12. Merlin engine v2.5.2 = v2.5.1 plus one fix.

- **Idle cards drop to P8 with the bounds guards on.** The guard reader polled every 50 ms forever, holding idle RTX
  3090s in P2: 446 W against 110 W in P8 on four cards. It now parks once the engine drains, after a final read of every
  counter, so no bounds check is lost. `VLLM_E2_GUARD_IDLE=0` restores the old behaviour. The same fix ships in
  Qwen3.8-27B v1.3.0. Thanks to the community member who reported it and proposed the first fix.
- No kernel, scheduler or numerics change; decode and prefill read as v2.5.1's within run-to-run noise.

## v2.5.1

2026-10-06. Flash-Next and Swift1.5 (tag `v2.5.1-swift1.5`). v2.5.0 was not published.

- **Agent follow-up turns resume from cache** on a 64-token grid instead of the 4,096-token block: median time to
  first token on follow-ups 0.96 s (v2.2.0: 1.56 s). With eight deep conversations at once, 92.5 % of follow-up prompt
  tokens came from cache.
- **KV pool 806,792 → 924,993 tokens (+15 %)**: six 131K or three 262K sessions resident with 0 preemptions.
- Cold prefill 5,219-5,533 tok/s from 10K to 261K, 1.0-2.8 % above v2.2.0; single-stream decode about 3 % below it.
- Transformers 5.18.0 (combining marks tokenize as upstream), up to 42 images per prompt at 4 MP each.
- GSM8K-200 198/200, structured output 80/80, the 262K needle found exactly.

## v2.2.0

2026-09-25.

- **Flash-Next** moves onto the public vLLM 0.30.0 tag plus 58 commits: a reliability release at parity with v2
  (greedy repeatability, any-rank fault detection, faster shutdown).
- **New: Swift1.5** (tag `v2.2.0-swift1.5`, 2026-09-28), the Merlin checkpoint with the 300 dense tensors Swift 1.5
  changed, its own FP8 KV scales and `xhigh` effort by default.
- **New: Qwen3.8-27B v1.0.0** (2026-09-27), the first 27B release: stock vLLM 0.30.0 plus five patches, two cards (pool
  794,351) or four (1,949,844), FP8 KV, MTP K=3, froggeric's chat template.

## v2.0.1

2026-09-18. The v2 shape plus one engine fix: structured output under concurrency.

## v2

2026-09-17. Flash-Next moves from four-way tensor parallel to **tensor parallel 2 × pipeline parallel 2** with expert
parallel, the FP8 n-gram table served over a host-mapped pull transport and the token embeddings moved into pinned
host memory. The KV pool grows from 342,912 to **806,792 tokens**: three full 262K sessions resident at once.
Single-stream decode 162 / 166 / 169 / 176 tok/s at 4K / 32K / 131K / 261K (parity with v1); prefill 10-26 % faster.

## v1

2026-09. The first Flash-Next release: vLLM 0.28.0 plus community commits and six patches, four-way tensor parallel with
expert parallel, MTP K=3, KV pool 342,912 tokens.

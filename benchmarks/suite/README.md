# Merlin bench suite

The two main benches behind the published bench cards (`benchmarks/2026-11-12/` and earlier), runnable
against any OpenAI-compatible Merlin server: the single-stream ladder (decode, cold prefill, time to
first token, the exact 262K row) and the needle panel. Same prompts, output caps, thinking settings,
validity rules and medians as the cards, so your figures can be read against the published ones.

Standard library only (Python 3.9 or newer). It sends requests and writes a report; it does not start,
configure or restart a server.

## Run it

```
python benchmarks/suite/run.py --base http://127.0.0.1:8000 --profile fn-quad-3090 --section decode
python benchmarks/suite/run.py --base http://127.0.0.1:8000 --profile 27b-tp4-3090 --all --out ./cards/27b-tp4
```

| option | meaning |
|---|---|
| `--section decode\|prefill\|ttft\|exact262\|needle` | one card section; repeat the option or comma-separate for several |
| `--all` | the five sections, in that order |
| `--profile ROW` | the launch row the server was started with (`fn-quad-3090`, `27b-tp4-3090`, ...). It picks the rows the card carries and labels the report; it configures nothing |
| `--base URL`, `--model ID`, `--api-key KEY` | the server (default `http://127.0.0.1:8000`, the first model of `/v1/models`) |
| `--effort LEVEL` | reasoning effort sent with thinking-on requests (default `medium`, the server default of the cards) |
| `--out DIR` | where the report goes (default `./bench-suite-<profile>-<utc>`) |

Exit code 0: every send valid. 3: some send was invalid (the report is written and lists why). 2: refused
(see below) or a bad argument.

Profile rows with a published card: `fn-quad-3090`, `fn-quad-20g-sm86` (ladder at 4K and 131K, needle panel
23 cases at 262K and 10 at 131K), `27b-tp4-3090`, `27b-tp2-3090` (ladder at 4K and 131K, needle panel 12 and 5),
`27b-tp1-3090` (ladder at 4K, 32K, 131K and 261K, needle panel 12 and 5). Another `fn-*` row uses the
Flash-Next rows, another `27b-*` row the one-card rows, any other name gets the full ladder and says so in the
report. Section by section:

| section | what it sends | cards' table |
|---|---|---|
| `decode` | per ladder depth, 3 cold sends with thinking on (up to 256 tokens) and 3 with thinking off (the answer-plus-summary task, natural stop, up to 1,024 tokens); each cell is the median of its valid sends | 1 |
| `exact262` | exactly 261,888 prompt tokens + 128 out, both thinking modes, 3 reps: rep 1 cold, reps 2 and 3 share its cache salt with a different needle code (so they reuse the prefix before it) | 2 |
| `prefill` | one cold send each at 10,240 and 102,400 nominal tokens, 8 tokens out. With `exact262` in the same run, its cold thinking-on send is the 262K row | 3 |
| `ttft` | 3 cold sends per ladder depth, 8 tokens out, thinking on: median time to the first token event | the TTFT column |
| `needle` | the needle panel, see below | quality panel |

## Output

`report.json` (every median unrounded, the protocol constants, the server and thinking facts), `report.md`
(the published cards' layout: sections 1 to 6 with the same table columns; only the sections that ran are
rendered) and `rows.jsonl` (one record per send: the unrounded token events, usage, finish reason, the
prompt's sha256, every validity problem). Figures are rounded only when `report.md` is rendered, from the raw
medians; every table figure can be recomputed from `rows.jsonl`.

## The protocol

- **Prompts.** A haystack of one repeated filler sentence with one needle (a code, once) 55 % in, then a
  question; plus the same small text image (it prints an order number) in every request. Nominal depths 4,096 / 32,768 / 131,072 /
  261,120 and prefill 10,240 / 102,400 are requested through a fixed table of builder sizes (the builder
  overshoots its nominal depth by a few per cent), so a depth lands at about 0.995 of its name on the
  reference tokenizer (4,046 prompt tokens for the 4K row). The prompt text is byte-identical to the cards'
  for the same nonce (`protocol.selftest()` checks it against a frozen hash).
- **Thinking.** On: `chat_template_kwargs = {"enable_thinking": true, "reasoning_effort": <effort>}`, every
  request, effort `medium` unless you change it. Off: `{"enable_thinking": false}`. The thinking-off ladder
  and exact-262K prompt asks for the answer lines and then about 300 words of prose, and a send is valid only
  if it stops by itself, writes at least 200 tokens and passes the content gate. **The report records what the
  responses showed**: how many thinking-on responses carried reasoning (a reasoning delta or a reasoning-token
  count) and how many thinking-off responses did, and warns when thinking-on showed none or thinking-off
  showed some.
- **Requests.** Temperature 0, streamed with `stream_options.include_usage`, `logprobs=true` with
  `top_logprobs=0`, and a fresh `cache_salt` per cold send (the server must honour it: a send that reports
  cached tokens is invalid).
- **Timing.** Client-side. A token event is a streamed chunk with at least one logprob entry. TTFT = request
  sent to the first event. tok/s = tokens after the first event / time from the first to the last event.
  ITL = the median gap between events; tokens/event = tokens / events (above 1 under MTP). Read tok/s with ITL
  and tokens/event.
- **Validity of a send.** HTTP 200; finish reason `stop` or `length`; at least two token events; strictly
  increasing arrivals; tokens in events equal `completion_tokens`; positive finite metrics; no prefix-cache hit
  on a salted cold send (the exact row's reps 2 and 3 are expected to hit and are reported, not failed); the
  thinking-off gates above.
- **Medians of 3.** Each ladder cell and the exact row are the median of 3 sends (valid ones; the table
  says `valid n/3`). Cold prefill is one send per depth, as on the cards.
- **Exact row.** The prompt is sized with the server's `/tokenize` plus the template overhead of a 4K probe
  (`usage.prompt_tokens` minus the tokenizer count) until `usage.prompt_tokens` is 261,888 on every rep;
  rows record `exact`.
- **Needle panel.** Single-occurrence needles: the code appears once in the prompt, 55 % into the haystack;
  an image in the prompt; thinking off; temperature 0 and a fixed seed; a fresh cache salt per request; one
  request at a time; 256 tokens out. Depths 262,144 and 131,072 (targets 258,644 and 130,672 prompt tokens),
  case counts from the profile. A case is valid when the reply is HTTP 200 with usage, the depth was reached
  (at least 98 % of the target) with the output cap inside it, and nothing was served from cache. The panel
  reports planned, completed-valid, completion yield, exact passes of the valid cases with a Wilson 95 %
  interval, the legacy pass (the code anywhere in the reply) and the incomplete count. Exact means the first
  `ANSWER1:` line equals the code after markdown and trailing punctuation are stripped.
- **Warm-up.** Two discarded sends before the first measured one: 8,192 tokens with 16 out, then a 4K prompt
  with 256 out (`--no-warm` skips them and the report says so).
- **Refusals.** The run stops with exit code 2 and writes no report when `/v1/models` cannot be read, the
  model is not served, the warm-up fails, or the stream lacks `usage` or the logprob entries (the card timing
  is defined on them). A section that cannot run (no `/tokenize` for the exact row and the needle panel, the
  server's context shorter than the depth) is skipped and named in the report's warnings; sections that ran
  are still reported.

## What differs from a lab run

The published figures were measured on a reference host with a method this suite does not reproduce. Expect
differences for these reasons, and do not read a small gap as a regression:

- **The server.** A card is measured on a fresh `merlin serve` boot per configuration with the shipped
  defaults, on a quiet host (4 x RTX 3090 at 220 W, P2P enabled). The suite measures whatever is running
  now: its build, power limit, clocks, co-tenants, thermal state and host load are yours. A single run has no
  repeat boot and no A/A band; MTP acceptance alone moves decode by several per cent between runs.
- **Build and engine facts.** The cards carry the wheel, the boot time, the KV pool and checks on the engine
  log (OOM and fatal lines, MTP acceptance). The suite sees only the API and records only what it returns.
- **Prompt sizes.** Ladder and prefill prompts are the same text for every server and are sized in the
  reference tokenizer's tokens. A checkpoint with another tokenizer counts a different number (the report
  shows the `prompt_tokens` the server counted), so a row is the same prompt, not the same token count.
- **Sizing in the exact row and the needle panel.** The suite sizes these prompts with the server's own
  `/tokenize` and a probe request; a lab run counted with the checkpoint's tokenizer and the image's
  placeholder tokens from its preprocessor config. The needle sits 55 % into the filler by length, not by token
  position. Counts land within about 0.1 % of the target.
- **Needle scoring.** The cards' panel was run as two halves on separate boots and scored as pairs for an
  A/B verdict, with replays of failures. The suite runs the panel once and reports exact passes with an
  interval; it makes no paired verdict. Its requests carry a seed but not the token-id echo a lab run
  stored.
- **Not included.** Agents at once, sessions held at once, 8 MP image rows, GSM8K and the 32K needle quality
  panel are separate benches, and so are comparisons with the previous release's column.
- **Boot index.** Ladder nonces carry a boot index (`--boot`, default 1, which reproduces the cards' prompts
  byte for byte). A different index changes the prompts and the report says so.

## Comparing with a card

Read a row against the card of the same checkpoint and layout (`benchmarks/2026-11-12/*-BENCH-CARD.md`).
Compare tok/s together with ITL and tokens/event, check `valid` is 3/3 and `prompt tokens` is the same,
and compare medians against the card's own spread, not against a hair: the cards say, for each figure,
how much the same row moves between identical runs.

## Smoke runs and tests

`--depths`, `--prefill-depths`, `--exact-target`, `--needle-counts DEPTH:N,...`, `--boot` and `--no-warm`
change the protocol for a quick check against a small model. A run that uses any of them is not a card run;
the report says so in its warnings.

Its tests (the real client and report code against a mock server on localhost, no GPU) live in the
development tree and are not shipped in this release.

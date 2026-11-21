# Agentic concurrency benchmark

How does a server hold up when several coding or research agents share it? This benchmark runs 1, 2, 4
and 8 agents at once (any list you like) against an OpenAI-compatible chat server and reports, per
workload and concurrency level:

- **output tok/s** and **total tok/s** (prompt + output), from the server's `usage` fields, over the
  level's wall time;
- **prefix-cache hit share**: cached prompt tokens / prompt tokens, from
  `usage.prompt_tokens_details.cached_tokens` (shown as `n/a` when the server does not report it);
- **task success rate**, checked by tests or an answer key, plus a partial-credit score;
- **mean and p95 step latency** (one step = one chat request) and **wall time**;
- optionally, **value per box-hour**, net value and margin, from prices *you* supply.

It writes `report.json` (every level, every episode and every step) and `report.md` (a summary table
in the same shape as the one in the GitHub issue that asked for this benchmark, plus a per-level table).

The client is a handful of Python files that use only the standard library (Python 3.11+). It runs
fully offline: every workload is generated from a seed or read from files bundled in `data/`.

## Quick start

Start any OpenAI-compatible server with tool calling enabled. For Merlin and vLLM that means a
tool-call parser for your model, for example:

```
vllm serve <model> --enable-auto-tool-choice --tool-call-parser <parser> --enable-prompt-tokens-details
```

`--enable-prompt-tokens-details` makes vLLM return `cached_tokens`; without it the prefix-hit column
reads `n/a`. Then, from the repository root:

```
python benchmarks/agentic/agentic_bench.py --self-check            # offline: checks the bundled tasks
python benchmarks/agentic/agentic_bench.py --base-url http://127.0.0.1:8000 --agents 1,2,4,8
```

The model defaults to the first one `/v1/models` lists (`--model` to choose). `--api-key` or
`$OPENAI_API_KEY` sets a bearer token. Results go to `agentic-results/<UTC time>/` (`--out` to choose).

To value the tokens, pass all four prices yourself (per million tokens, and the box's cost per hour,
in one currency). There are no defaults:

```
python benchmarks/agentic/agentic_bench.py --base-url http://127.0.0.1:8000 \
    --price-in <p> --price-cached <p> --price-out <p> --box-cost <c> --label "<your hardware>"
```

Useful options (`--help` lists all):

| option | default | meaning |
|---|---|---|
| `--workloads` | `tool_loop,repo_fix,research` | which workloads, in order |
| `--agents` | `1,2,4,8` | concurrency levels |
| `--episodes-per-agent` | 2 | tasks each agent runs back to back in a level |
| `--max-steps` | 40 | chat requests per episode at most |
| `--max-tokens` | 4096 | `max_tokens` per request |
| `--temperature` / `--seed` | 0 / 0 | sampling, and the seed for task choice and generated workspaces |
| `--n-funcs` | 6 | stub functions per tool-loop workspace |
| `--system-pad-words` | 0 | deterministic filler added to the shared system prompt, to emulate the long system prompts of real agent harnesses |
| `--extra-body` | none | JSON merged into every request, e.g. `'{"chat_template_kwargs": {"enable_thinking": false}}'` |
| `--no-level-salt` | off | see *Prefix cache* below |
| `--tool-timeout` | 30 | seconds per sandboxed test run |
| `--keep-workspaces` | off | keep each agent's temporary directory for inspection |

## Protocol

1. One untimed warm-up request (`--warmup`).
2. For each workload, for each concurrency level N: N agent threads start together on a barrier. Each
   runs `--episodes-per-agent` episodes back to back. Agent *a*'s episode *e* is task number
   `a * episodes_per_agent + e`, so the same seed gives the same tasks on every run.
3. An episode is a plain tool-calling loop: send the conversation and the workload's tools to
   `/v1/chat/completions` (not streamed), run every tool call in the reply in order, append the results
   as `tool` messages, repeat. It ends when the model calls the finishing tool (`finish` or
   `submit_answer`), after `--max-steps` requests, or after `--max-nudges` replies in a row with no tool
   call (each such reply gets a one-line reminder). A server error ends the episode and is counted.
4. The level's clock stops when the last agent finishes. The checkers run after that, outside the
   timed window.
5. Tokens are taken only from the server's `usage` of each response; nothing is estimated with a local
   tokenizer.

Assistant replies are sent back with their `content` and `tool_calls`; any separate reasoning field the
server returns is not sent back, which matches what most agent clients do.

## Workloads

**`tool_loop`, synthetic tool loop.** A generated workspace holds `funcs.py` with N stub functions
whose docstrings specify small, exact behaviours (affine maps, clamps, character counts, sliding-window
maxima, a Caesar shift and so on, with seeded constants). For each function the agent must implement
it, add `tests/test_<name>.py`, and run the suite. It measures a long, regular tool-call loop with a
growing context: many short steps, each re-sending the whole conversation. Success: hidden checker
tests (generated from the same seed, written into the workspace only after the episode) pass for every
function, and a test file exists for each.

**`repo_fix`, repository bug fixes.** Four tiny bundled Python repositories (`data/repo_tasks/`), each
with a bug report and failing unittest tests: word counting, interval merging, a shopping cart with
Decimal rounding, and calendar arithmetic. The agent reads the code, edits it and runs the tests. It
measures explore-edit-test loops with more reading and fewer, larger edits. Success: the original tests
(restored over whatever the agent left in `tests/`, so editing the tests does not help) all pass. The
tasks were written for this benchmark and are not taken from any public bug dataset; each ships a
reference fix in `solution/` that the agent never sees and `--self-check` uses.

**`research`, read-heavy research.** Twelve questions about a bundled corpus of ten short documents
(`data/corpus/`) describing a fictional river valley and its railway, written for this benchmark so
the answers cannot come from a model's training data. The agent has `search` (keyword search over the
corpus) and `fetch` (one document by id) tools, both answered locally, and must combine facts across
documents, sometimes with arithmetic. It measures read-heavy steps: large tool results, little output.
Success: the submitted answer contains the answer key's value as a whole token (`data/research_questions.json`).

## Reading the numbers

- **Total tok/s** counts every prompt token the server was sent, including the cached ones; agent
  workloads re-send the whole conversation each step, so it is dominated by prompt tokens and is
  only meaningful next to the prefix-hit share. **Output tok/s** is the generated tokens.
- **Prefix hit** comes only from the server. If only some responses report `cached_tokens`, the share is
  computed over those, and the report notes the coverage.
- **Value per box-hour** = (uncached input x input price + cached input x cached price + output x output
  price) / wall hours. Net = value - box cost; margin = net / value. A missing cached price bills cached
  tokens at the input price; without `cached_tokens` from the server all input is billed as uncached.
  The prices are whatever you choose to compare against: the benchmark ships none.
- **Avg context** in the summary is the mean prompt tokens per request over that workload's levels.

## Prefix cache between levels

All levels reuse the same tasks, so a later level could hit the prefix cache left by an earlier one and
look better than it is. By default the first line of the system prompt is a run tag that names the
workload, seed and level, which keeps a level from reusing an earlier level's cache while agents within
one level still share their common system prompt and tools. `--no-level-salt` drops the tag.

## Sandbox and safety

Tool calls never reach a shell. The file tools (`list_files`, `read_file`, `write_file`,
`replace_in_file`) act only inside the episode's own temporary directory. Every path is checked:
absolute paths, drive letters, `~`, `..` components, NUL bytes and anything that resolves outside the
workspace (symlinks included) are refused, and the refusal goes back to the model as the tool result.
Writes are capped at 256 KiB per file.

The only tool that runs model-written code is `run_tests`, which runs the fixed command "unittest
discovery in `tests/`" in a child Python with:

- isolated mode (`-I -B`): no user site-packages, no `PYTHON*` environment variables, no bytecode files;
- the workspace as its working directory, a minimal environment (`HOME` and `TMPDIR` inside the
  workspace), and no stdin;
- a wall-clock timeout (`--tool-timeout`), after which its whole process group is killed;
- on POSIX, CPU-time, file-size and address-space limits;
- an audit hook that refuses socket connections and name lookups, process creation, `ctypes` library
  loading, and file writes, renames and deletions outside the workspace.

The `fetch` and `search` tools of the research workload read only the bundled corpus held in memory;
they never touch the network or the filesystem beyond `data/corpus/`.

This is a guard against accidents (a model that writes `rm` into a test, or a test that tries the
network), not a security boundary against a model trying to escape: Python audit hooks can be bypassed
by determined native code. When the model or the server is untrusted, run the benchmark inside a
container or VM with no network.

## Caveats

- The workloads are small on purpose so that anyone can run them offline. Their contexts are shorter
  than those of a full coding agent; use `--system-pad-words` (and more `--n-funcs` for the tool loop)
  to approach a given context length, and report the avg context with your numbers.
- Success depends on the model, its chat template and the server's tool-call parser as much as on
  the server's speed. A run where most episodes end in `no_tool_call` usually means the tool-call parser
  is missing or wrong.
- Throughput includes the time agents spend in tools (test runs take a fraction of a second each), as
  a real agent would; it does not include the checkers.
- Temperature 0 and a fixed seed make the task set identical between runs, but batching on the server
  can still change the generated tokens, so success and token counts vary a little from run to run.
  Run each configuration more than once before comparing two servers.
- The answer check for the research workload is lenient (the key value must appear as a whole token in
  a short answer); it guards against wrong answers, not against an answer that lists several values.

## Files

| path | contents |
|---|---|
| `agentic_bench.py` | command line, level runner |
| `agentic_loop.py` | standard-library chat client and the agent loop |
| `agentic_workloads.py` | the three workloads, tool schemas, checkers, `--self-check` |
| `agentic_sandbox.py` | per-agent workspace, path policy, sandboxed test runner |
| `agentic_metrics.py` | metric arithmetic |
| `agentic_report.py` | JSON report and markdown tables |
| `data/` | repo tasks, research corpus and answer key |
| `conftest.py` | keeps a repository-wide pytest run out of the repo tasks' deliberately failing tests |

Its tests (CPU only; a mock server stands in for the model) live in the development tree and are not
shipped in this release.

# Connect a client

When the server is up, `merlin serve` prints one line with everything a client needs:

```text
merlin: Ready: http://127.0.0.1:8100/v1 · model qwen38-27b · context 262144 · key in /home/you/merlin-engine/.merlin/api-key (started in 3m12s)
merlin:   Anthropic-compatible endpoint: http://127.0.0.1:8100/v1/messages (same model and key)
```

The entries below use those values as placeholders:

| Placeholder | From the Ready line |
|---|---|
| `HOST:PORT` | the URL (`<this-host>` means a network bind: use the server's address) |
| `MODEL` | `model` |
| `CONTEXT` | `context` |
| `KEY` | the key file's contents (`cat .merlin/api-key`); any non-empty string when the line says `no API key` |

Two settings matter in every client:

- **Context length:** set the client's context window to `CONTEXT`. Most clients assume a smaller default and trim
  history early, or a larger one and send requests the server rejects.
- **Reasoning effort:** leave it at **medium** (the served default) on models whose chat template has effort levels;
  Nemotron Super VL and Ornith 1.5 have none. A client that sends its own effort overrides it.

Client configuration formats change between releases; if an entry below no longer matches your client, its own
documentation for "OpenAI-compatible provider" (or "Anthropic-compatible base URL") has the current field names.

## OpenAI-compatible clients

```bash
export OPENAI_BASE_URL=http://HOST:PORT/v1
export OPENAI_API_KEY=KEY
```

```python
from openai import OpenAI

client = OpenAI()  # reads both variables
print(client.chat.completions.create(model="MODEL", messages=[{"role": "user", "content": "Say hello."}])
      .choices[0].message.content)
```

## Anthropic-compatible clients

```bash
export ANTHROPIC_BASE_URL=http://HOST:PORT
export ANTHROPIC_AUTH_TOKEN=KEY
export ANTHROPIC_MODEL=MODEL
export ANTHROPIC_DEFAULT_HAIKU_MODEL=MODEL  # its background calls go to the same model
# then start your Anthropic-compatible client
```

The base URL has no `/v1`: the client adds `/v1/messages` itself.

## opencode

`opencode.json` (in the project, or `~/.config/opencode/`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "merlin": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Merlin",
      "options": { "baseURL": "http://HOST:PORT/v1", "apiKey": "KEY" },
      "models": { "MODEL": { "name": "MODEL", "limit": { "context": CONTEXT, "output": 32768 } } }
    }
  }
}
```

Then pick `merlin/MODEL` in opencode's model list.

## Cline and Roo Code

In the extension's settings: **API Provider** `OpenAI Compatible`, **Base URL** `http://HOST:PORT/v1`, **API Key**
`KEY`, **Model ID** `MODEL`, and under the model's settings **Context Window Size** `CONTEXT`.

## Aider

```bash
aider --model openai/MODEL --openai-api-base http://HOST:PORT/v1 --openai-api-key KEY
```

Aider learns the context window from `.aider.model.metadata.json` (in the project or your home directory):

```json
{ "openai/MODEL": { "max_input_tokens": CONTEXT, "max_tokens": 32768, "litellm_provider": "openai", "mode": "chat" } }
```

## Continue

`~/.continue/config.yaml`:

```yaml
models:
  - name: Merlin
    provider: openai
    model: MODEL
    apiBase: http://HOST:PORT/v1
    apiKey: KEY
    defaultCompletionOptions:
      contextLength: CONTEXT
```

## When the Ready line does not come

`/health` answers only once the server is ready: the weights are loaded, the kernels are warmed up and the CUDA graphs
are captured. Until then connections are refused, and `merlin serve` prints a `starting...` line every minute. A first
start compiles kernels and takes longer than later ones. If it never comes, `merlin doctor --report` writes one file
(preflight, the serve plan, the server log's last lines, secrets redacted) to attach to a bug report.

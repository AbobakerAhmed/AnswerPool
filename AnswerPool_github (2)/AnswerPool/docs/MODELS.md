# Model registry and backends

## Models

`python models.py` lists the configured models; `MODEL=<key>` in the driver
resolves the backend, base URL and per-provider flags from that table, and a
raw model id works too (the backend is guessed from its prefix).

| key | backend | api id | key needed |
|---|---|---|---|
| `opus-5`, `sonnet-5`, `fable-5.1`, `haiku-4.5` | `anthropic` | `claude-opus-5`, ... | `ANTHROPIC_API_KEY` |
| `gpt-5`, `gpt-5.2`, `gpt-5.2-chat` | `openai` | `gpt-5.2`, `gpt-5.2-chat-latest` | `OPENAI_API_KEY` |
| `kimi-k3`, `kimi-k2.6` | `openai` | `kimi-k3` at `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` (from platform.kimi.ai) |
| `gemini-flash` | `api` | `gemini-2.5-flash` | `GOOGLE_API_KEY` or `VERTEX_KEY` (free tier available) |
| `gpt-4o-free`, `gpt-4.1-free` | `openai` | via GitHub Models | `GITHUB_TOKEN` -- **free, no card** |
| `llama-vision-free` | `openai` | via GitHub Models | `GITHUB_TOKEN` -- **free, no card** |
| `qwen3-8b`, `qwen2.5-vl-7b` | `vllm` | local weights | none -- **free**, needs a GPU |

### Known vLLM issue: Qwen2.5-VL rope_scaling conflict

Some vision checkpoints (Qwen2.5-VL and relatives) ship a `config.json`
whose `rope_scaling` has both the legacy `type` field and the modern
`rope_type` field; current vLLM validates that only one is present and
raises `ValidationError: Found conflicts between 'rope_type=...' (modern
field) and 'type=...' (legacy field)`. `run_matching.py`'s vLLM path patches
this automatically -- it fetches (or reuses) the model's config, strips the
legacy key if both are present, and loads from the patched local snapshot.
No flag needed; look for a `[rope-patch] ...` line in the output when it fires.

### Free options (no payment anywhere)

None of the true current frontier models (Opus 5, GPT-5.2, Kimi K3) have a
free API tier -- all three are pay-per-token with no allowance, which is the
wall a $0-balance Moonshot account or a fresh Anthropic key hits immediately.
Two paths avoid spending anything:

- ~~GitHub Models~~ was permanently retired 2026-07-30 and no longer works
  at any endpoint (`models.github.ai/inference` now returns 410 for every
  call). It is not in the registry any more; do not re-add it.
- **Gemini free tier** (`gemini-flash`): free, no card, via Google AI
  Studio (aistudio.google.com -> Get API key). Flash-class only -- Pro
  models are paid-only. Google's terms allow free-tier prompts to be used
  to improve their models; mention this if it matters for the writeup.
- **Local weights on Kaggle's free GPU** (`qwen2.5-vl-7b` via `--backend
  vllm`): zero API cost at all, uses the free T4 GPU quota instead. Slower
  to start (loads weights) and per-request, but no rate limit and no data
  leaves the notebook.

```bash
MODEL=opus-5   ./run_benchmarks.sh gpqa_diamond
MODEL=gpt-5.2  ./run_benchmarks.sh mmmu
MODEL=kimi-k3  ./run_benchmarks.sh hellaswag
```

Three things change when you evaluate an API model rather than local weights,
and the paper's protocol depends on all three:

* **No grammar-constrained decoding.** Section 3.5 gets 100 percent format
  compliance from xgrammar, which only exists on the vLLM path. On an API the
  parser can fail, so `unparsed slots` is reported separately from wrong
  answers and must be quoted alongside any accuracy number. Check it is near
  zero before comparing an API model against a local one.
* **Reasoning models.** GPT-5 and Kimi K3 reject `temperature` and rename the
  token cap; `run_matching.py` discovers both from the first error and
  remembers them. Their reasoning burns the output budget, so the registry
  sets `--max-out 4096` and `--reasoning-effort minimal`, which is the closest
  match to the paper's thinking-disabled protocol. Without it a starved pass
  returns empty text that scores as a wrong answer. Claude models are called
  with extended thinking off for the same reason.
* **Snapshots, not fixed checkpoints.** As Section 3.5 notes, served aliases
  carry no determinism guarantee, so API results are a dated snapshot; keep
  the raw outputs (they are stored) and let the local open-weight runs carry
  any claim that must outlive a provider update.

Provider keys are not interchangeable, so each has its own variable and
several can coexist on one machine: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`MOONSHOT_API_KEY` (or `KIMI_API_KEY`), `OPENROUTER_API_KEY`,
`GOOGLE_API_KEY`. The openai backend picks the one matching `OPENAI_BASE_URL`
and prints which variable it used; a local `vllm serve` needs none.
`python models.py --resolve <model>` reports the variable a model needs.

Vision: Claude, GPT-5 and Kimi K3 all accept images, so any of them can run
MMMU, MMMU-Pro, MathVista and ScienceQA. `models.py` marks which entries are
vision-capable.


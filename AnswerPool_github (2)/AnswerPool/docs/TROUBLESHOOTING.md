# Troubleshooting

Real bugs hit while running vision-language models through vLLM on Kaggle T4
GPUs, in the order they compound. Fix ONE at a time, with `--no-deps` where
shown, then restart the session before retrying -- do not pin several
packages "just in case" in one pass, which is what turned individual
10-minute fixes into a multi-hour chase during development.

## Troubleshooting the vLLM backend on Kaggle T4

Four separate, genuine upstream compatibility bugs were hit and fixed while
building this pipeline. All four are now handled automatically by the code
(the rope-scaling one) or documented here (the rest, since they are pip
environment issues outside this codebase's control). If a run fails, check
the error against this list before changing anything broader.

**Rule for all of these: fix ONE thing at a time, with `--no-deps` where
shown, then `Session -> Restart Session`, then retry.** Do not pin several
packages "just in case" in one pass -- that is what turned a 10-minute fix
into a multi-hour one during development. Also: **any pip install after this
point should be followed by a restart before you trust a version check** --
Python keeps the old module in memory even after files on disk change.

| Error message contains... | Cause | Fix |
|---|---|---|
| `Found conflicts between 'rope_type=...' and 'type=...'` | Qwen2.5-VL config quirk | Nothing to do -- `run_matching.py` patches this automatically now. If you still see it, your code is stale; re-upload the zip. |
| `AttributeError: ... has no attribute all_special_tokens_extended` | Installed `transformers` is too new for this vLLM build | `!pip install "transformers<5,>=4.55" --no-deps` |
| `Numba needs NumPy 2.2 or less. Got NumPy X.X` | Installed `numpy` is too new for `numba` (a vLLM sub-dependency) | `!pip install "numpy<2.3" --force-reinstall --no-deps` |
| `AttributeError: module 'torch._dynamo' has no attribute 'utils'` at `import vllm` | torch itself got corrupted by a previous `--force-reinstall` of something else | `!pip uninstall -y torch torchvision torchaudio vllm transformers numpy tokenizers`, then `!pip cache purge`, then plain `!pip install vllm==0.11.0` with NOTHING else pinned. Let it resolve torch on its own. |
| `ImportError: libcudart.so.13: cannot open shared object file` | An unpinned `pip install vllm` (no version) pulled a CUDA-13 torch build; Kaggle's T4 image only has CUDA 12.x runtime libs | Pin the version again: `!pip install vllm==0.11.0` (not unpinned) |
| Hugging Face download stuck at `0%` / `0.00B` for minutes | `hf_transfer`'s fast downloader occasionally hangs on Kaggle's network for large repos | Already disabled by default in cell 2 (`HF_HUB_ENABLE_HF_TRANSFER=0`). If it still hangs, interrupt and retry with `force_download=True` and `max_workers=2`. |
| `prompt too long` / `skipped: prompt too long` | `--max-len` is smaller than a pooled prompt's actual token count (text + all embedded images) | Raise `--max-len` (try 32768 for N=5 pools with several images each); MMMU-Pro's worst observed case was ~23,700 tokens. |
| A `.jsonl` file is missing that you built earlier in this session | Kaggle did a full/factory-reset restart, which wipes `/kaggle/working` entirely (not just `/kaggle/temp`) | Re-run cells 1-2, then rebuild the missing file(s) -- see cell 7. Consider `File -> Save Version` once a run succeeds, so you have a checkpoint that survives a reset. |
| `no API key for base_url=...` right after switching `MODEL` | A previous model's `OPENAI_BASE_URL` leaked into this run | Already fixed in cell 6 (always sets or clears it). Re-run cell 6 after changing `MODEL`, don't just edit the variable and skip straight to running. |

## Additional bugs found running larger/newer vision models (Qwen3-VL)

| Error message contains... | Cause | Fix |
|---|---|---|
| `RuntimeError: Failed to create unquantized linear weights` / generic OOM at model load | The model's base weights + KV cache genuinely exceed available VRAM | Lower `--max-len`, add `--gpu-util 0.80`, or use a smaller model variant (e.g. Qwen3-VL-4B instead of -8B) |
| OOM whose traceback runs through `torch._inductor` / `cuda_graph.py` / `cuda_piecewise_backend.py`, NOT through weight loading | vLLM's CUDA-graph compilation step needs scratch memory ON TOP of the model+cache; this can tip an otherwise-fitting model over the edge on a smaller GPU | Add `--enforce-eager` (skips graph capture, trades some speed for a smaller peak footprint) |
| `RuntimeError: Engine core initialization failed. Failed core proc(s): {}` with no visible root cause above it | The actual crash (often OOM) happened in a worker subprocess whose output got truncated in the notebook's captured log | Scroll to the very top of the cell's output, before the traceback -- the real error is usually there. Re-run with more visible output if needed. |
| Same crash reproduces identically even after a clean `pip uninstall` + reinstall | Not a leftover-state problem -- likely a genuine upstream `vllm`/`torch` version incompatibility for this exact model | Search the exact error text on `github.com/vllm-project/vllm/issues` before re-guessing pins; a plain, unpinned `pip install vllm==<version>` (no `torch`/`transformers` override) is more likely to resolve a mutually-compatible set than manual pinning |
| A fix "isn't taking effect" despite re-uploading the dataset | Two datasets with different content are both attached; the code search picks by path length or arbitrary order, not recency | `!find /kaggle/input -name "run_matching.py" -exec grep -c "<the fix>" {} \; -exec stat -c '%y %n' {} \;` to see every copy directly. `kaggle_answerpool.py`'s cell 1 sorts by modification time and warns when multiple candidates exist, but always verify with a direct `grep` against `/kaggle/input` rather than trusting the copy alone. |
| Changing `--votes` on `verify_wellformed.py` triggers a full, re-billed re-run instead of reusing prior results | The checkpoint file path is derived from `--detail`'s filename; pointing `--detail` at a new file also starts a fresh (empty) checkpoint | Use `--recompute-from <old-detail>.partial.jsonl` to re-tally votes at zero API cost, instead of changing `--detail` |


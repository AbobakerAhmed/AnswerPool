# Running on Kaggle

## Running on Kaggle

`python kaggle_answerpool.py` writes `kaggle_answerpool.ipynb`, a notebook that
does the whole setup: finds the code, points the HF cache at `/kaggle/temp`,
reads keys from Kaggle Secrets, runs the fit check, builds the arms and scores
them. Import it under Create -> Notebook -> File -> Import Notebook.

Four things to set in the notebook's right-hand panel before running:

* **Internet: On** (phone-verified account) — `pip` and the Hub need it.
* **Accelerator**: `None` for an API model (12 h sessions, no GPU quota),
  `GPU T4 x2` for a local model (9 h sessions, 30 h/week).
* **Secrets**: `HF_TOKEN` (GPQA is gated), plus `GOOGLE_API_KEY` or
  `OPENAI_API_KEY`.
* **Input**: this repo, uploaded as a Kaggle Dataset, or cloned from GitHub.

Kaggle specifics the code now handles:

* **T4s have no bfloat16.** `--dtype auto` (the default) detects compute
  capability below 8.0 and switches vLLM to float16; without it the engine
  refuses to start on the free GPUs.
* **`/kaggle/working` is capped at 20 GB and is committed with the notebook.**
  Keep `HF_HOME=/kaggle/temp/hf` so model weights and dataset caches stay out
  of it; only built `.jsonl` files, question images and results belong there.
* **Sessions expire.** Results carry a prompt hash, so re-running an arm in a
  later session resumes: add the previous notebook's output as an Input, copy
  the `.results.jsonl` files back, re-run. Use `--limit` on the first pass.
* **Two T4s hold a 7B vision model** in float16 with `--tp 2 --max-len 8192`;
  larger vision models do not fit.


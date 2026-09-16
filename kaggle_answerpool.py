#!/usr/bin/env python
# ============================================================================
# kaggle_answerpool.py — build the notebook that runs this pipeline on Kaggle.
#
#   python kaggle_answerpool.py --out kaggle_answerpool.ipynb
#
# Upload the .ipynb to Kaggle (Create -> Notebook -> File -> Import Notebook),
# or copy the cells by hand. See the README section "Running on Kaggle".
# ============================================================================
import argparse
import json
import re

CELLS = [
    ("md", """# Answer Pooling on Kaggle

Runs the de-saturation transform on GPQA Diamond, HellaSwag, MMMU, MMMU-Pro,
MathVista and ScienceQA.

**Before running, in the right-hand panel:**

1. **Settings -> Internet: On** (needs a phone-verified account). Without it,
   neither `pip` nor the Hugging Face datasets can be reached.
2. **Settings -> Accelerator**: `None` if you evaluate an API model (12 h
   sessions, no GPU quota), `GPU T4 x2` for a local model via vLLM (9 h
   sessions, 30 h/week).
3. **Add-ons -> Secrets**: add `HF_TOKEN` (GPQA is gated; accept its licence
   on the Hub first) and, for API models, `GOOGLE_API_KEY` or `OPENAI_API_KEY`.
4. **Input -> Add Input**: the dataset holding this code. Upload
   `AnswerPool_benchmarks.zip` at kaggle.com/datasets -> New Dataset, then
   attach it here. Cloning from a private GitHub repo does not work on Kaggle:
   git has no credentials and fails asking for a username."""),

    ("code", """# --- 1. get the code -------------------------------------------------------
# Kaggle auto-extracts an uploaded zip, and where the files land depends on how
# the zip was built: /kaggle/input/<ds>/, /kaggle/input/<ds>/ap/, or a zip left
# sitting inside the dataset. Search instead of guessing one layout.
import os, glob, shutil, sys

MARKER   = "build_matching.py"
CODE     = "/kaggle/working/answerpool"
# Set this if the search below misses your dataset. It is the folder that
# directly contains build_matching.py, e.g.
#   "/kaggle/input/datasets/<user>/<slug>/ap"
CODE_SRC = ""

def _find(root, depth=8):
    hits = []
    if not os.path.isdir(root):
        return hits
    for d, dirs, files in os.walk(root):
        if d[len(root):].count(os.sep) > depth:
            dirs[:] = []
            continue
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        if MARKER in files:
            hits.append(d)
    # Sort by how recently the marker file was modified, NOT by path length.
    # Sorting by length picked whichever dataset slug happened to be
    # alphabetically/numerically shorter -- with two datasets attached (e.g.
    # an old one and a freshly re-uploaded one), that can silently pick the
    # STALE one forever regardless of which was actually uploaded last, and
    # every downstream "the fix isn't there" report traces back to this.
    return sorted(hits, key=lambda d: os.path.getmtime(os.path.join(d, MARKER)),
                  reverse=True)

_hits = _find("/kaggle/input")
if len(_hits) > 1:
    print("!! multiple datasets attached that all contain this project's code:")
    for d in _hits:
        t = os.path.getmtime(os.path.join(d, MARKER))
        print(f"   {d}  (code last modified {__import__('datetime').datetime.fromtimestamp(t)})")
    print(f"   using the most recently modified one: {_hits[0]}")
    print("   if that's wrong, remove the stale dataset from this notebook's "
          "Input panel, or set CODE_SRC above explicitly.")
src = CODE_SRC or (_hits[0] if _hits else None)
if src and not os.path.exists(os.path.join(src, MARKER)):
    raise SystemExit(f"CODE_SRC={src!r} does not contain {MARKER}")
if src is None:                      # a .zip inside the dataset also works
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        shutil.rmtree("/kaggle/working/code_tmp", ignore_errors=True)
        shutil.unpack_archive(z, "/kaggle/working/code_tmp")
        src = next(iter(_find("/kaggle/working/code_tmp")), None)
        if src:
            print("unpacked", z)
            break

if src is None:
    print(sorted(glob.glob("/kaggle/input/*")) or "  (nothing attached)")
    raise SystemExit("\\n".join([
        "Could not find the code.",
        "Add it as a Kaggle Dataset:",
        "  1. kaggle.com/datasets -> New Dataset -> upload AnswerPool_benchmarks.zip",
        "  2. in this notebook: Input -> Add Input -> your dataset",
        "  3. re-run this cell",
        "A private GitHub repo cannot be cloned here: git has no credentials and",
        "prompts for a username, which is the error you saw. If the repo is private,",
        "either upload the zip as above, or add a token secret and clone",
        "https://<TOKEN>@github.com/<user>/<repo>."]))

os.makedirs(CODE, exist_ok=True)
for f in glob.glob(src + "/*.py") + glob.glob(src + "/*.sh") + glob.glob(src + "/*.txt") \
        + glob.glob(src + "/*.md"):
    shutil.copy(f, CODE)
os.chdir(CODE)
sys.path.insert(0, CODE)
print("code from:", src)
print(sorted(os.listdir(CODE)))

# Sanity check the fixes below actually landed -- a stale dataset upload is
# the single most common cause of "I already fixed this" showing up again.
checks = {
    "build_matching.py": "min-options",
    "run_matching.py": "patch_rope_conflict",
    "models.py": "gemini-3.6-flash",
}
print("")
print("fix check (all should say YES):")
for fname, needle in checks.items():
    body = open(fname.strip(), encoding="utf-8").read() if os.path.exists(fname.strip()) else ""
    ok = "YES" if needle in body else "NO -- reupload the zip"
    print("  " + fname.ljust(22) + " has " + repr(needle) + ": " + ok)"""),

    ("code", """# --- 2. caches and secrets -------------------------------------------------
# /kaggle/working is capped at ~20 GB and is saved with the notebook; dataset
# and model caches must NOT go there or the commit will fail. /kaggle/temp is
# scratch and disappears at session end, which is what we want for caches.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
# hf_transfer (the accelerated Rust downloader) has been observed to hang at
# 0% indefinitely on Kaggle's network for large repos (15GB+ vision models);
# plain HTTP is slower per-byte but actually starts and completes reliably.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.makedirs("/kaggle/temp/hf", exist_ok=True)

OUT = "/kaggle/working/runs"        # results + images: these DO persist
os.makedirs(OUT, exist_ok=True)

WANT = ["HF_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY",
        "KIMI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "OPENAI_BASE_URL"]
found, missing = [], []
try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    for name in WANT:
        try:
            v = sec.get_secret(name)
            if v:
                os.environ[name] = v
                found.append(name)
            else:
                missing.append(name)
        except Exception:
            missing.append(name)
except ImportError:
    print("not on Kaggle; set the environment variables yourself")

print("secrets loaded :", ", ".join(found) or "NONE")
print("not available  :", ", ".join(missing))
if not found:
    # A silent failure here is what makes cell 8 die with "set <KEY>" much
    # later, after the datasets are downloaded and the arms are built.
    for line in [
            "",
            "!! No secrets reached this kernel. Two steps are needed, and the",
            "   second is the one people miss:",
            "     1. Add-ons -> Secrets -> Add secret (label = ANTHROPIC_API_KEY)",
            "     2. tick the ATTACH checkbox for THIS notebook in that same",
            "        dialog, then re-run this cell",
            "   Without step 2 the secret exists on your account but the kernel",
            "   cannot read it."]:
        print(line)

!df -h /kaggle/working /kaggle/temp | cat"""),

    ("code", """# --- 3. dependencies -------------------------------------------------------
# datasets/pillow are usually present but often outdated on the Kaggle image.
!pip install -q -U "datasets>=2.14" "pillow>=10" 2>&1 | tail -2

# API backends (pick the one you need; both are small)
!pip install -q google-genai openai anthropic 2>&1 | tail -2

# Local models only. vLLM reinstalls torch, takes ~10 min, and the session
# must be restarted afterwards (Run -> Restart & clear cell outputs), then
# re-run cells 1 and 2. Leave commented if you use an API model.
# !pip install -q vllm==0.11.0 2>&1 | tail -3"""),

    ("code", """# --- 4. sanity check: does the code run at all? ----------------------------
# Offline, no network, no keys: builds every arm on synthetic data and scores it.
!python smoke_test.py 2>&1 | tail -15"""),

    ("code", """# --- 5. choose the run -----------------------------------------------------
BENCH   = "scienceqa"      # gpqa_diamond | hellaswag | mmmu | mmmu_pro | mathvista | scienceqa
MODEL   = "Qwen/Qwen3-VL-8B-Instruct"  # newer generation than Qwen2.5-VL,
                           # needs GPU T4 x2 + vLLM (cell 3). Run cell 5b
                           # right after this to check vLLM actually
                           # supports it before spending GPU time.
                           # FREE alternatives: gemini-flash (GOOGLE_API_KEY,
                           # no card, no GPU needed) or qwen2.5-vl-7b (local,
                           # confirmed working -- see project history).
                           # Paid: opus-5 | sonnet-5 | gpt-5.2 | kimi-k3.
                           # GitHub Models (gpt-4o-free etc.) is permanently
                           # retired 2026-07-30 -- do not use those keys.
# models.py --shell emits shell assignments (quoted); parse them with shlex
import shlex, subprocess as _sp
_out = _sp.run([sys.executable, "models.py", "--resolve", MODEL, "--shell"],
               capture_output=True, text=True, check=True).stdout
_r = {}
for _line in _out.splitlines():
    if "=" in _line:
        _k, _v = _line.split("=", 1)
        _r[_k] = (shlex.split(_v) or [""])[0]
BACKEND     = _r["BACKEND"]
MODEL_ID    = _r["MODEL_ID"]
MODEL_EXTRA = _r.get("MODEL_EXTRA", "")
if _r.get("MODEL_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = _r["MODEL_BASE_URL"]
KEY_ENV = _r.get("MODEL_KEY_ENV", "")
print(f"{MODEL} -> backend={BACKEND} id={MODEL_ID} extra={MODEL_EXTRA!r}")
print(f"base_url = {os.environ.get('OPENAI_BASE_URL', '(none -- default endpoint)')}")
if KEY_ENV:
    ok = "set" if os.environ.get(KEY_ENV) else "NOT SET -- add it in Add-ons -> Secrets"
    print(f"needs secret {KEY_ENV}: {ok}")
N       = 5                # questions per group (3 for mmmu / mathvista)
WITHHOLD= 0.4              # fraction of golds deleted -> unanswerable questions
LIMIT   = 200              # records per arm; 0 = all. Keep it small on a first pass.
MIN_OPT = 4                # drop items with fewer options. ScienceQA is 53%
                           # two-option items: those give the MCQ baseline a
                           # 0.50 floor and add one distractor to the pool, so
                           # the comparison is against a coin flip. 4 matches
                           # the QuALITY/GPQA construction; 0 disables.

# Kaggle sessions are 9 h (GPU) / 12 h (CPU). Every result file stores a hash
# of its prompt, so re-running the same cell in a later session RESUMES rather
# than redoing work -- provided the results stay in /kaggle/working and you
# add the previous notebook's output as an Input dataset."""),

    ("code", """# --- 5b. new-model compatibility check (vLLM only) --------------------------
# A newer model architecture (e.g. Qwen3-VL) may not be recognised by the
# vllm==0.11.0 pinned in cell 3 -- it may predate the model's release. Check
# in seconds here rather than after a model-load failure mid-run.
if BACKEND == "vllm":
    try:
        from vllm.transformers_utils.config import get_config
        cfg = get_config(MODEL_ID, trust_remote_code=True)
        arch = getattr(cfg, "architectures", ["?"])[0]
        print("model architecture:", arch)
        from vllm.model_executor.models.registry import ModelRegistry
        supported = arch in ModelRegistry.get_supported_archs()
        print("supported by this vllm build:", supported)
        if not supported:
            print()
            print("!! NOT SUPPORTED by vllm==0.11.0. Options:")
            print("   1. !pip install -U vllm  then Session -> Restart Session")
            print("      (may reopen the torch/transformers version fights from")
            print("      earlier -- see the troubleshooting table further down)")
            print("   2. use qwen2.5-vl-7b instead, which is confirmed working")
    except Exception as e:
        print(f"could not check ({type(e).__name__}: {e}); vLLM will report "
              f"this definitively when it actually tries to load the model")

"""),

    ("code", """# --- 6. does this benchmark fit the method? --------------------------------
# Measures groups retained, W1 duplicate-gold loss, pool size, option-type
# mixing. Read the flags before trusting any score. (Downloads the dataset.)
!python check_fit.py --dataset {BENCH} --n {N} --min-options {MIN_OPT}"""),

    ("code", """# --- 7. build the arms -----------------------------------------------------
VIS = BENCH in ("mmmu", "mmmu_pro", "mathvista", "scienceqa")
flags = f"--dataset {BENCH} --n {N} --distractors"
if MIN_OPT:
    flags += f" --min-options {MIN_OPT}"

!python build_matching.py {flags} --withhold {WITHHOLD} --out {OUT}/{BENCH}_match.jsonl
!python build_matching.py --dataset {BENCH} --min-options {MIN_OPT} --arm mcq --mirror {OUT}/{BENCH}_match.jsonl --out {OUT}/{BENCH}_mcq.jsonl
!python build_matching.py {flags} --arm choices_only --out {OUT}/{BENCH}_co.jsonl

# visual benchmarks: the image is the passage, so the closed-book control is
# the image-withheld arm. Run it on 3 models, then filter_blind.py, then
# rebuild with --exclude before reporting matching numbers.
if VIS:
    !python build_matching.py {flags} --arm mcq_no_passage --out {OUT}/{BENCH}_noimg.jsonl

!du -sh {OUT} | cat"""),

    ("code", """# --- 8. run --------------------------------------------------------------
# vLLM extras: --dtype auto picks float16 on T4 (no bfloat16 there), --tp 2
# uses both T4s, --max-len keeps the KV cache inside 16 GB.
extra = MODEL_EXTRA or ("--tp 2 --gpu-util 0.90" if BACKEND == "vllm" else "--workers 4")

for arm in (["match", "mcq", "co"] + (["noimg"] if VIS else [])):
    f = f"{OUT}/{BENCH}_{arm}.jsonl"
    print("=" * 70, "\\n", arm)
    !python run_matching.py --in {f} --model {MODEL_ID} --backend {BACKEND} --limit {LIMIT} {extra}"""),

    ("code", """# --- 9. compare ------------------------------------------------------------
!python compare.py --glob "{OUT}/{BENCH}_*.results.jsonl\""""),    ("md", """## Troubleshooting the vLLM backend on Kaggle T4

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
| `no API key for base_url=...` right after switching `MODEL` | A previous model's `OPENAI_BASE_URL` leaked into this run | Already fixed in cell 6 (always sets or clears it). Re-run cell 6 after changing `MODEL`, don't just edit the variable and skip straight to running. |"""),



    ("code", """# --- 10. (optional) the FULL report matrix ---------------------------------
# 4 frontier vision models x (4 visual benchmarks + GPQA Diamond), five arms
# each (match p=0.4, match0 no-withhold, mirror MCQ, choices-only, and the
# image-withheld MCQ on visual benchmarks), then Tables 3/4/7/8 as .md/.tex.
# This is the expensive one: every model needs its key in Secrets, GPQA needs
# HF_TOKEN. Start with LIMIT=50 to see cost; 0 = all records.
REPORT_MODELS = "opus-5 gpt-5.2 kimi-k3 gemini-2.5-pro"
REPORT_BENCH  = "gpqa_diamond mmmu mmmu_pro mathvista scienceqa"
REPORT_LIMIT  = 50
RUN_FULL_REPORT = False          # set True to run

if RUN_FULL_REPORT:
    !MODELS="{REPORT_MODELS}" BENCH="{REPORT_BENCH}" LIMIT={REPORT_LIMIT} OUT={OUT} bash run_report.sh
    from IPython.display import Markdown, display
    display(Markdown(open(f"{OUT}/report_tables.md").read()))"""),

    ("md", """## Notes

**Resuming across sessions.** Results land in `/kaggle/working/runs` and are
saved as the notebook's output. To continue in a new session, add that output
as an Input dataset, copy the `.results.jsonl` files back into `runs/`, and
re-run cell 8: matching prompt hashes are reused, changed ones are discarded.

**Disk.** MMMU's validation split with images is several GB; keep `HF_HOME`
on `/kaggle/temp`. Only the built `.jsonl` files, the question images and the
results belong in `/kaggle/working` (20 GB cap, and a commit fails if exceeded).

**Which models.** `python models.py` lists the keys. Claude, GPT-5 and Kimi K3
all take images, so on a CPU-only kernel (12 h, no GPU quota) any of them can
run the visual benchmarks. Note that API backends have no grammar-constrained
decoding, so check `unparsed slots` in the run output before comparing against
a local model.

Visual benchmarks need a vision model on every backend.
On two T4s, 7B-class vision models in float16 fit (`Qwen/Qwen2.5-VL-7B-Instruct`
with `--tp 2 --max-len 8192`); larger ones do not. Text benchmarks
(GPQA Diamond, HellaSwag) run 7B to 8B models comfortably.

**If vLLM fails to start.** Lower `--gpu-util` to 0.85, lower `--max-len`, or
switch the accelerator to `GPU T4 x2` and pass `--tp 2`. `--dtype float16`
is forced automatically on T4 but can be set explicitly."""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle_answerpool.ipynb")
    a = ap.parse_args()
    cells = []
    for i, (kind, text) in enumerate(CELLS):
        lines = text.split("\n")
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append(dict(cell_type="markdown", id=f"c{i}", metadata={}, source=src))
        else:
            cells.append(dict(cell_type="code", id=f"c{i}", metadata={},
                              execution_count=None, outputs=[], source=src))
    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                            language="python"),
                            language_info=dict(name="python")))
    # Guard against the escaping bug this generator is prone to: a "\\n"
    # written with one backslash too few becomes a real newline inside the
    # cell's source and the cell dies with SyntaxError in the notebook, far
    # from here. Parse every code cell before writing the file.
    import ast
    bad = 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        body = "".join(l for l in c["source"]
                       if not l.lstrip().startswith(("!", "%")))
        # a magic inside an if/for leaves an empty block; pad it
        body = re.sub(r"(?m)^(\s*)(if|for|while|else|elif|try)\b(.*):\s*$",
                      r"\1\2\3:\n\1    pass", body)
        try:
            ast.parse(body)
        except SyntaxError as e:
            bad += 1
            print(f"  CELL {i} IS BROKEN: {e.msg} (line {e.lineno})")
            for n, l in enumerate(body.splitlines()[max(0, e.lineno - 3):e.lineno], 
                                  start=max(1, e.lineno - 2)):
                print(f"    {n:3} {l}")
    if bad:
        raise SystemExit(f"{bad} cell(s) would fail in the notebook; fix and re-run")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {a.out} ({len(cells)} cells, all code cells parse)")


if __name__ == "__main__":
    main()

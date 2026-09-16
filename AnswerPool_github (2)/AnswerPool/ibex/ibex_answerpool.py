#!/usr/bin/env python
# ============================================================================
# ibex_answerpool.py — build ibex_answerpool.ipynb, the Ibex counterpart of
# kaggle_answerpool.py. Simpler than the Kaggle version because Ibex has no
# per-session wipe, no dataset-search-in-/kaggle/input machinery, and no
# Kaggle Secrets: the code lives at a fixed path you control, the env is
# built once by setup_env.sh, and keys are read from ~/.answerpool_env.
#
#   python ibex_answerpool.py --out ibex_answerpool.ipynb
#
# Prerequisites (run once, outside the notebook):
#   bash setup_env.sh                    # builds ~/venvs/answerpool
#   git clone/rsync this project to ~/answerpool
#   sbatch jupyter_job.sh                # launches Jupyter, see that file
# ============================================================================
import argparse
import ast
import json
import re

CELLS = [
    ("md", """# Answer Pooling on Ibex

Same pipeline as the Kaggle notebook, run here because it needs to survive
longer than a Kaggle session and because you already have Ibex access.

**Before running:** `setup_env.sh` must have been run once (builds a
persistent venv), and this notebook must be opened via `jupyter_job.sh`'s
tunnel (see that file's header) so it's running with GPUs attached."""),

    ("code", """# --- 1. environment check ---------------------------------------------------
import os, subprocess, sys

CODE_DIR = os.path.expanduser("~/answerpool")
os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import torch, vllm, transformers, numpy
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print()
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                      "--format=csv"], capture_output=True, text=True).stdout)

# fix check -- confirms this checkout has the fixes the Kaggle session needed
for fname, needle in [("build_matching.py", "min-options"),
                      ("run_matching.py", "patch_rope_conflict"),
                      ("models.py", "gemini-3.6-flash")]:
    body = open(fname, encoding="utf-8").read()
    print(f"{fname:22} has {needle!r}: {'YES' if needle in body else 'NO -- update the checkout'}")"""),

    ("code", """# --- 2. keys and cache -------------------------------------------------------
# Ibex persists /home and $SCRATCH across sessions, unlike Kaggle -- set this
# up once and it stays. Put keys in ~/.answerpool_env as plain KEY=value
# lines (chmod 600 it) rather than typing them into a cell every session.
env_file = os.path.expanduser("~/.answerpool_env")
if os.path.exists(env_file):
    for line in open(env_file):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v
    print("loaded keys from", env_file)
else:
    print(f"no {env_file} found. Create it with lines like:")
    print("  HF_TOKEN=hf_...")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    print("  MOONSHOT_API_KEY=sk-...")
    print("(only needed if you use a gated dataset or an API backend; vLLM "
          "with a public model like Qwen2.5-VL needs none of these)")

os.environ.setdefault("HF_HOME", os.path.expanduser(os.environ.get("SCRATCH", "~/scratch") + "/hf_cache"))
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
print("HF_HOME:", os.environ["HF_HOME"])"""),

    ("code", """# --- 3. sanity check: does the code run at all? ------------------------------
!python smoke_test.py 2>&1 | tail -15"""),

    ("code", """# --- 4. choose the run --------------------------------------------------------
BENCH   = "mmmu_pro"
MODEL   = "Qwen/Qwen3-VL-8B-Instruct"   # newer generation than the 2.5-VL
                                        # run this project already has results
                                        # for -- see cell right after this one
                                        # for a compatibility check first.
                                        # or a models.py key, e.g. "opus-5"
N       = 5
MIN_OPT = 4
WITHHOLD = 0.4
LIMIT   = 0            # 0 = full dataset. Ibex jobs don't randomly reset, so
                       # there is much less reason to sample here than on Kaggle.
OUT     = "runs/" + BENCH
os.makedirs(OUT, exist_ok=True)

import subprocess, shlex
r = subprocess.run([sys.executable, "models.py", "--resolve", MODEL, "--shell"],
                   capture_output=True, text=True)
_r = {}
for line in r.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        _r[k] = (shlex.split(v) or [""])[0]
BACKEND     = _r.get("BACKEND", "vllm")
MODEL_ID    = _r.get("MODEL_ID", MODEL)
MODEL_EXTRA = _r.get("MODEL_EXTRA", "")
if _r.get("MODEL_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = _r["MODEL_BASE_URL"]
else:
    os.environ.pop("OPENAI_BASE_URL", None)
print(f"{MODEL} -> backend={BACKEND} id={MODEL_ID} extra={MODEL_EXTRA!r}")"""),

    ("code", """# --- 4b. new-model compatibility check -----------------------------------------
# Qwen3-VL is a newer architecture than Qwen2.5-VL. This project is pinned to
# vllm==0.11.0 (see setup_env.sh), which predates Qwen3-VL's release -- it
# may not recognise the architecture at all. Check in seconds rather than
# finding out after a 10-minute model load on a full batch job.
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
            print("!! NOT SUPPORTED. Options:")
            print("   1. pip install -U vllm  (may reintroduce the torch/CUDA")
            print("      conflicts fixed in setup_env.sh -- test on a short")
            print("      interactive session first: srun --gpus=1 --time=00:15:00 --pty bash)")
            print("   2. use Qwen2.5-VL-7B-Instruct instead, which is confirmed working")
    except Exception as e:
        print(f"could not check ({type(e).__name__}: {e}); vLLM will tell you "
              f"definitively when run_matching.py actually loads the model")

"""),

    ("code", """# --- 5. does this benchmark fit the method? -----------------------------------
!python check_fit.py --dataset {BENCH} --n {N} --min-options {MIN_OPT}"""),

    ("code", """# --- 6. build the arms ---------------------------------------------------------
VIS = BENCH in ("mmmu", "mmmu_pro", "mathvista", "scienceqa")
flags = f"--dataset {BENCH} --n {N} --distractors --min-options {MIN_OPT}"

!python build_matching.py {flags} --withhold {WITHHOLD} --out {OUT}/{BENCH}_match.jsonl
!python build_matching.py --dataset {BENCH} --min-options {MIN_OPT} --arm mcq --mirror {OUT}/{BENCH}_match.jsonl --out {OUT}/{BENCH}_mcq.jsonl
!python build_matching.py {flags} --arm choices_only --out {OUT}/{BENCH}_co.jsonl
!python build_matching.py {flags} --withhold {WITHHOLD} --out {OUT}/{BENCH}_wh.jsonl
if VIS:
    !python build_matching.py --dataset {BENCH} --min-options {MIN_OPT} --arm mcq_no_passage --mirror {OUT}/{BENCH}_match.jsonl --out {OUT}/{BENCH}_noimg.jsonl

!du -sh {OUT}"""),

    ("code", """# --- 7. run --------------------------------------------------------------------
extra = MODEL_EXTRA or ("--tp 2 --max-len 32768" if BACKEND == "vllm" else "--workers 4")
arms = ["match", "mcq", "co", "wh"] + (["noimg"] if VIS else [])

for arm in arms:
    f = f"{OUT}/{BENCH}_{arm}.jsonl"
    print("=" * 60, "\\n", arm)
    !python run_matching.py --in {f} --model {MODEL_ID} --backend {BACKEND} --limit {LIMIT} {extra}"""),

    ("code", """# --- 8. report -------------------------------------------------------------
!python report_tables.py --bench {BENCH} --dir {OUT} --out {OUT}/report
print()
!python consolidate_report.py --bench {BENCH} --dir {OUT}"""),

    ("md", """## Notes for Ibex specifically

**This session doesn't reset.** Unlike Kaggle, nothing here needs redoing
next time -- `runs/`, the venv, and `$SCRATCH/hf_cache` all persist. Re-open
via `jupyter_job.sh` and continue from cell 1.

**If a run needs longer than the SLURM job's `--time` limit**, either raise
`--time` in `jupyter_job.sh` and resubmit, or split arms across separate
`sbatch` jobs using `run_mmmu_pro.sbatch` instead of this notebook for the
long unattended runs -- keep the notebook for interactive exploration and
the batch script for the actual full-scale run.

**Troubleshooting:** the same version-conflict table from the Kaggle
notebook applies if you ever `pip install` something inside this venv and
it breaks vLLM -- see that notebook's troubleshooting section, or re-run
`setup_env.sh` into a fresh venv directory as a clean reset."""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ibex_answerpool.ipynb")
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

    bad = 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        body = "".join(l for l in c["source"] if not l.lstrip().startswith(("!", "%")))
        body = re.sub(r"(?m)^(\s*)(if|for|while|else|elif|try)\b(.*):\s*$",
                      r"\1\2\3:\n\1    pass", body)
        try:
            ast.parse(body)
        except SyntaxError as e:
            bad += 1
            print(f"  CELL {i} IS BROKEN: {e.msg} (line {e.lineno})")
    if bad:
        raise SystemExit(f"{bad} cell(s) would fail; fix and re-run")

    nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                            language="python"),
                            language_info=dict(name="python")))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {a.out} ({len(cells)} cells, all code cells parse)")


if __name__ == "__main__":
    main()

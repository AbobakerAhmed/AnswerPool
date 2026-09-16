# Running on Ibex (KAUST HPC)

Ibex has no click-to-run notebook — you launch a Jupyter server as a SLURM
job on a compute node, then tunnel your browser into it over SSH. Unlike
Kaggle, nothing here gets wiped between sessions: the environment, built
`.jsonl` files, and results all persist in your home directory / `$SCRATCH`.

## One-time setup

```bash
bash ibex/setup_env.sh          # builds a persistent venv at ~/venvs/answerpool
git clone <this-repo> ~/answerpool
```

`setup_env.sh` installs `vllm==0.11.0` plain — no manual `torch`/
`transformers`/`numpy` pins alongside it. Every attempt to pre-pin those on
Kaggle triggered a `torch`/CUDA corruption that took hours to diagnose (see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)); let vLLM's own resolver pick a
consistent set.

## Launching Jupyter

```bash
sbatch ibex/jupyter_job.sh
tail -f jupyter_<jobid>.log      # wait for the ssh -L ... line
```

Copy the printed `ssh -L ...` command into a **new terminal on your own
laptop**, run it, leave it open. Open the printed `http://127.0.0.1:<port>/...`
URL in your browser, then open `notebooks/ibex_template.ipynb`.

## Or skip the notebook for long unattended runs

```bash
sbatch ibex/run_mmmu_pro.sbatch
squeue -u $USER
```

A plain `sbatch` job has no SSH tunnel to keep alive, so it's more robust
for a multi-hour full-scale run than the notebook path — use the notebook
for interactive exploration, the batch script for the run you want to walk
away from.

## Reproducibility note

`build_matching.py --seed 0` (the default) produces byte-identical groups
and pools regardless of platform, since it only depends on the dataset and
the seed. Model *outputs* are expected to match Kaggle's results closely but
not necessarily bit-for-bit — different GPU architectures (T4 vs. A100) can
shift floating-point rounding on borderline token choices. This is the same
caveat the original paper states for its own local-model reproducibility
(Section 3.5): "reproduce to within batching-order float noise."

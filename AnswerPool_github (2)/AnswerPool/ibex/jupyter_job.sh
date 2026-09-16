#!/bin/bash
#SBATCH --job-name=answerpool_jupyter
#SBATCH --time=06:00:00
#SBATCH --gpus=2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=jupyter_%j.log
# ============================================================================
# jupyter_job.sh — launches Jupyter Lab on an Ibex GPU compute node, with
# this project's environment and paths already set up. Run once per work
# session (unlike Kaggle, nothing here gets wiped between sessions).
#
#   1. sbatch jupyter_job.sh
#   2. tail -f jupyter_<jobid>.log   -- wait for the ssh -L ... line
#   3. paste that ssh command into a NEW terminal on your own laptop, run it,
#      leave it open
#   4. open the http://127.0.0.1:<port>/... URL from the log in your browser
#   5. in Jupyter, open ibex_answerpool.ipynb
#
# Run setup_env.sh once, ever, before the first use of this script.
# ============================================================================
set -euo pipefail

ENV_DIR="${ENV_DIR:-$HOME/venvs/answerpool}"
CODE_DIR="${CODE_DIR:-$HOME/answerpool}"
HF_CACHE="${HF_CACHE:-$SCRATCH/hf_cache}"

source "$ENV_DIR/bin/activate"
pip show jupyterlab >/dev/null 2>&1 || pip install jupyterlab

export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p "$HF_CACHE"

export XDG_RUNTIME_DIR=/tmp
node=$(hostname -s)
port=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

echo "================================================================"
echo " On YOUR OWN laptop, in a NEW terminal, run this and leave it open:"
echo
echo "   ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} $USER@glogin.ibex.kaust.edu.sa"
echo
echo " Then open the http://127.0.0.1:${port}/... URL printed below in your browser."
echo "================================================================"

cd "$CODE_DIR"
jupyter lab --no-browser --port="${port}" --ip="${node}.ibex.kaust.edu.sa"

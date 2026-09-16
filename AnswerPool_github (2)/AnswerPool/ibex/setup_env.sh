#!/bin/bash
# ============================================================================
# setup_env.sh — ONE-TIME environment setup on Ibex. Run this once, ever,
# on a login node (it's just pip installs, no GPU needed):
#
#   bash setup_env.sh
#
# This exists because the Kaggle session in this project's history hit FOUR
# separate genuine version-compatibility bugs between vllm/torch/transformers/
# numpy, each costing a full session restart to fix. The exact combination
# that was finally confirmed working end-to-end is pinned here so Ibex never
# has to rediscover it. Unlike Kaggle, this environment PERSISTS: do this
# once, then every job script below just activates it.
# ============================================================================
set -euo pipefail

ENV_DIR="${1:-$HOME/venvs/answerpool}"
echo "creating persistent venv at: $ENV_DIR"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install -U pip

# Install vLLM FIRST and PLAIN -- no manual torch/transformers/numpy pin
# alongside it. On Kaggle, every attempt to pre-pin those broke something
# else; letting vllm's own resolver pick a consistent set is what worked.
pip install vllm==0.11.0

# The ONE follow-up fix that was needed after that on Kaggle: numba (a vllm
# sub-dependency) rejects newer numpy. Pin narrowly, with --no-deps so it
# cannot drag anything else along with it.
pip install "numpy<2.3" --no-deps --force-reinstall || true

# Record exactly what got installed, so a future "it worked before" question
# has a real answer instead of a guess.
pip freeze > "$ENV_DIR/frozen-requirements.txt"
echo "recorded: $ENV_DIR/frozen-requirements.txt"

python - <<'PY'
import torch, vllm, transformers, numpy
print()
print("verify -- these are the versions the rest of this project assumes:")
print("  torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("  vllm:", vllm.__version__)
print("  transformers:", transformers.__version__)
print("  numpy:", numpy.__version__)
PY

echo
echo "done. Activate in any job with:  source $ENV_DIR/bin/activate"

#!/usr/bin/env bash
# ============================================================================
# run_benchmarks.sh — build, run and score the main benchmarks in one go:
#   GPQA Diamond · HellaSwag · MMMU · MMMU-Pro · MathVista · ScienceQA
#
# For each benchmark it builds three arms (matching with 40% withheld golds,
# the mirror-paired MCQ baseline, and the choices-only control), runs them
# against MODEL, and prints the comparison table.
#
#   MODEL=opus-5 ./run_benchmarks.sh
#   MODEL=gpt-5.2 ./run_benchmarks.sh mmmu mathvista
#   MODEL=kimi-k3 ./run_benchmarks.sh hellaswag
#   MODEL=Qwen/Qwen2.5-VL-7B-Instruct ./run_benchmarks.sh scienceqa
#
# MODEL takes a key from models.py (opus-5, sonnet-5, fable-5.1, gpt-5.2,
# kimi-k3, gemini-flash, ...) or a raw model id; the backend, base URL and
# per-provider flags come from that registry. `python models.py` lists them.
#   ./run_benchmarks.sh mmmu_pro10          # MMMU-Pro 10-option, N=3 M=24
#
# Fit to the paper's method (see README "Does each benchmark fit"): GPQA
# Diamond and ScienceQA fit; HellaSwag and MMMU need the choices-only and
# closed-book screens; MMMU-Pro "vision" is NOT poolable; MathVista is a poor fit.
#
# Env:
#   MODEL      model name (required)
#   BACKEND    api | openai | vllm            (default api)
#   BENCH      space list; or pass as args    (default: all five)
#   N          questions per group            (default 5; 3 for mmmu/mathvista,
#                                              whose topics are small)
#   WITHHOLD   fraction of golds withheld     (default 0.4)
#   LIMIT      records per run, 0 = all       (default 0)
#   EXTRA_RUN  extra args for run_matching.py (e.g. "--rpm 60 --workers 4")
#   OUT        output directory               (default runs/)
#
# Credentials: GOOGLE_API_KEY or VERTEX_KEY (api); OPENAI_API_KEY and optional
# OPENAI_BASE_URL (openai); HF_TOKEN for GPQA (gated on the Hub).
# Visual benchmarks need a vision-capable model on any backend.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

: "${MODEL:?set MODEL=<model name>}"
# resolve MODEL through the registry unless BACKEND is set explicitly
eval "$(python models.py --resolve "$MODEL" --shell)"
BACKEND="${BACKEND:-$BACKEND}"
MODEL="$MODEL_ID"
# always set or clear: a stale OPENAI_BASE_URL from a previous ./run of this
# same shell session would silently redirect this model's calls elsewhere
if [[ -n "$MODEL_BASE_URL" ]]; then export OPENAI_BASE_URL="$MODEL_BASE_URL"
else unset OPENAI_BASE_URL; fi
EXTRA_RUN="${EXTRA_RUN:-$MODEL_EXTRA}"
echo "model=$MODEL backend=$BACKEND base_url=${OPENAI_BASE_URL:-(none -- default endpoint)} extra='$EXTRA_RUN'"
WITHHOLD="${WITHHOLD:-0.4}"
LIMIT="${LIMIT:-0}"
OUT="${OUT:-runs}"
BENCH="${*:-${BENCH:-gpqa_diamond hellaswag mmmu mmmu_pro mathvista scienceqa}}"
mkdir -p "$OUT"

build_flags() {
  case "$1" in
    gpqa_diamond) echo "--dataset gpqa_diamond --n ${N:-5} --distractors" ;;
    hellaswag)    echo "--dataset hellaswag --n ${N:-5} --distractors" ;;
    mmmu)         echo "--dataset mmmu --n ${N:-3} --distractors" ;;
    mmmu_pro)     echo "--dataset mmmu_pro --mmmu-pro-config \"${MMMU_PRO_CONFIG:-standard (4 options)}\" --n ${N:-5} --distractors" ;;
    mmmu_pro10)   # 10-option items: keep 7 of 9 distractors at N=3 (M=24), the
                  # MMLU-Pro recipe of the paper. Keeping only 3 inverts the result.
                  echo "--dataset mmmu_pro --mmmu-pro-config \"standard (10 options)\" --n 3 --max-distractors 7 --distractors" ;;
    mathvista)    echo "--dataset mathvista --n ${N:-3} --distractors" ;;
    scienceqa)    echo "--dataset scienceqa --n ${N:-5} --distractors" ;;
    *) echo "unknown benchmark $1" >&2; exit 1 ;;
  esac
}

run() {
  local f=$1
  python run_matching.py --in "$f" --model "$MODEL" --backend "$BACKEND" \
    --limit "$LIMIT" $EXTRA_RUN
}

for b in $BENCH; do
  echo; echo "################ $b ################"
  flags=$(build_flags "$b")
  ds=$(echo "$flags" | sed -E 's/.*--dataset ([^ ]+).*/\1/')
  m="$OUT/${b}_match.jsonl"
  # 1. matching arm with withheld golds (accuracy + confabulation)
  eval python build_matching.py $flags --withhold "$WITHHOLD" --out "$m"
  # 2. mirror-paired MCQ baseline: same questions, original options
  cfg=$(echo "$flags" | grep -o -- '--mmmu-pro-config "[^"]*"' || true)
  eval python build_matching.py --dataset "$ds" $cfg --arm mcq --mirror "$m" --out "$OUT/${b}_mcq.jsonl"
  # 3. choices-only control: pool alone, no questions, no images
  eval python build_matching.py $flags --arm choices_only --out "$OUT/${b}_co.jsonl"

  # 4. visual benchmarks: image-withheld MCQ, the closed-book control. Run it
  #    on 3 models, then filter_blind.py -> --exclude to drop items solvable
  #    without the image before trusting any matching number (paper 3.5).
  if [[ "$ds" =~ ^(mmmu|mmmu_pro|mathvista|scienceqa)$ ]]; then
    eval python build_matching.py $flags --arm mcq_no_passage --out "$OUT/${b}_noimg.jsonl"
    run "$OUT/${b}_noimg.jsonl"
  fi

  run "$m"
  run "$OUT/${b}_mcq.jsonl"
  run "$OUT/${b}_co.jsonl"

  echo; echo "---- $b: comparison ----"
  python compare.py --glob "$OUT/${b}_*.results.jsonl" || true
done

echo; echo "done -> $OUT/"

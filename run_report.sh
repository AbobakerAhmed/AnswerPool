#!/usr/bin/env bash
# ============================================================================
# run_report.sh — the report's full matrix on the new benchmarks.
#
#   4 frontier vision models x (4 visual benchmarks + GPQA Diamond),
#   5 arms each, then report_tables.py writes Tables 3/4/7/8.
#
# Arms per benchmark (all item-paired to the same groups):
#   match    matching, withhold p=0.4            Table 4 (FA, false abst., strict)
#   match0   matching, no withhold (mirrored)    Table 3 / 7 (exact assignment)
#   mcq      mirror-paired MCQ                   MCQ per-q, group-scored MCQ
#   co       choices-only (mirrored)             alpha, Proposition 1
#   noimg    image-withheld MCQ (visual only)    closed-book rate, filter_blind
#
#   MODELS="opus-5 gpt-5.2 kimi-k3 gemini-2.5-pro" ./run_report.sh
#   MODELS="opus-5" BENCH="scienceqa gpqa_diamond" LIMIT=50 ./run_report.sh
#
# Env: MODELS (models.py keys), BENCH, LIMIT (records per arm, 0=all),
#      WITHHOLD (0.4), OUT (runs/), SKIP_RUN=1 (only build + tables)
# Keys per model: ANTHROPIC_API_KEY, OPENAI_API_KEY, MOONSHOT_API_KEY,
#      GOOGLE_API_KEY; HF_TOKEN for GPQA.
#
# The benchmark recipes below are the ones that fit the method (README,
# "Does each benchmark fit"): MMMU-Pro standard config only, never vision;
# ScienceQA at >=4 options; MathVista grouped by source with >=4 options.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODELS="${MODELS:-opus-5 gpt-5.2 kimi-k3 gemini-2.5-pro}"
BENCH="${BENCH:-gpqa_diamond mmmu mmmu_pro mathvista scienceqa}"
LIMIT="${LIMIT:-0}"
WITHHOLD="${WITHHOLD:-0.4}"
OUT="${OUT:-runs}"
SKIP_RUN="${SKIP_RUN:-0}"

recipe() {
  case "$1" in
    gpqa_diamond) echo "--dataset gpqa_diamond --n 5 --distractors" ;;
    mmmu)         echo "--dataset mmmu --n 3 --distractors --min-options 4" ;;
    mmmu_pro)     echo "--dataset mmmu_pro --mmmu-pro-config \"standard (4 options)\" --n 5 --distractors" ;;
    mmmu_pro10)   echo "--dataset mmmu_pro --mmmu-pro-config \"standard (10 options)\" --n 3 --max-distractors 7 --distractors" ;;
    mathvista)    echo "--dataset mathvista --n 3 --distractors --min-options 4 --group-by-visual source" ;;
    scienceqa)    echo "--dataset scienceqa --n 5 --distractors --min-options 4" ;;
    hellaswag)    echo "--dataset hellaswag --n 5 --distractors" ;;
    *) echo "unknown benchmark $1" >&2; exit 1 ;;
  esac
}
is_visual() { [[ "$1" =~ ^(mmmu|mmmu_pro|mmmu_pro10|mathvista|scienceqa)$ ]]; }

for b in $BENCH; do
  d="$OUT/$b"; mkdir -p "$d"
  flags=$(recipe "$b")
  ds=$(echo "$flags" | sed -E 's/.*--dataset ([^ ]+).*/\1/')
  cfg=$(echo "$flags" | grep -o -- '--mmmu-pro-config "[^"]*"' || true)
  mo=$(echo "$flags" | grep -o -- '--min-options [0-9]+' || true)
  echo; echo "################ build $b ################"
  if [[ ! -f "$d/${b}_match.jsonl" ]]; then
    eval python build_matching.py $flags --withhold "$WITHHOLD" --out "$d/${b}_match.jsonl"
    eval python build_matching.py $flags --withhold 0 --mirror "$d/${b}_match.jsonl" --out "$d/${b}_match0.jsonl"
    eval python build_matching.py --dataset "$ds" $cfg $mo --arm mcq --mirror "$d/${b}_match.jsonl" --out "$d/${b}_mcq.jsonl"
    eval python build_matching.py $flags --arm choices_only --mirror "$d/${b}_match.jsonl" --out "$d/${b}_co.jsonl"
    if is_visual "$b"; then
      eval python build_matching.py --dataset "$ds" $cfg $mo --arm mcq_no_passage --mirror "$d/${b}_match.jsonl" --out "$d/${b}_noimg.jsonl"
    fi
  else
    echo "already built: $d/${b}_match.jsonl (delete to rebuild)"
  fi
  python check_fit.py --dataset "$ds" $(echo "$flags" | grep -o -- '--n [0-9]+') $mo 2>/dev/null | tail -6 || true

  [[ "$SKIP_RUN" == "1" ]] && continue
  for mk in $MODELS; do
    eval "$(python models.py --resolve "$mk" --shell)"
    [[ -n "$MODEL_BASE_URL" ]] && export OPENAI_BASE_URL="$MODEL_BASE_URL" || unset OPENAI_BASE_URL
    echo; echo "======== $b x $mk ($MODEL_ID via $BACKEND) ========"
    arms="match match0 mcq co"; is_visual "$b" && arms="$arms noimg"
    for arm in $arms; do
      f="$d/${b}_${arm}.jsonl"
      python run_matching.py --in "$f" --model "$MODEL_ID" --backend "$BACKEND" \
        --limit "$LIMIT" --out "$d/${b}_${arm}.${MODEL_ID//\//_}.results.jsonl" $MODEL_EXTRA \
        || echo "!! $b/$arm/$mk failed (see above); continuing"
    done
  done
done

echo; echo "################ tables ################"
python report_tables.py --all --dir "$OUT" --out "$OUT/report_tables"
echo "-> $OUT/report_tables.md and .tex"

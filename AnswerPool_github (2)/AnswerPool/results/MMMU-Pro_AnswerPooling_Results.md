# MMMU-Pro Results — Qwen2.5-VL-7B-Instruct vs. Qwen3-VL-4B-Instruct

**Dataset:** MMMU-Pro, `standard (4 options)` config, N=5 questions pooled per
group, M=15–20 candidates, grouped by subject (30 subjects). Full 256-group
build (no sampling).

**Method:** Answer Pooling (Barebood, Ahmed, Bu Subait, Eltahir, Hussain,
*"Your Benchmark Is Not Saturated: Reviving Multiple-Choice Evaluation with
Answer Pooling"*), extended in this repo to MMMU-Pro — no shared passage,
grouped by subject exactly as the paper groups MMLU-Pro/GPQA/C-Eval, with
each question keeping its own individual image.

**Raw execution:** [`notebooks/mmmu_pro_reproduction.ipynb`](../notebooks/mmmu_pro_reproduction.ipynb)
— real cells and real outputs from the Kaggle session that produced these
numbers, not re-run for this document.

---

## Table 3-style — De-saturation (11-group exclusion, 245 groups)

| | MCQ | Group-scored MCQ (×5) | Matching (exact) |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 0.4629 | 0.0898 | 0.0490 |
| Qwen3-VL-4B-Instruct | 0.4849 | 0.0735 | 0.0653 |

## Table 4-style — Withhold arm, p=0.4 (M=18)

| | Accuracy | False answer ↓ | False abstention | Strict ↑ |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 0.3184 | **0.9469** | 0.0503 | 0.0000 |
| Qwen3-VL-4B-Instruct | 0.3306 | **0.9878** | 0.0095 | 0.0000 |

## Table 7-style — Full comparison

| | Choices-only | Matching (per-q) | MCQ (per-q) | Match. exact | Grp. MCQ | Image-withheld (per-q) |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | 0.0410 | 0.3184 | 0.4629 | 0.0490 | 0.0898 | 0.3429 |
| Qwen3-VL-4B-Instruct | 0.0443 | 0.3306 | 0.4849 | 0.0653 | 0.0735 | 0.3567 |

## Statistical tests (paired bootstrap, groups resampled, 95% CI)

| Test | Estimate | 95% CI | Significant? |
|---|---|---|---|
| False-answer rate, match arm | −0.0388 | [−0.0592, −0.0184] | **Yes** |
| False-answer rate, withhold arm | −0.0408 | [−0.0612, −0.0204] | **Yes** |
| Accuracy, choices-only | −0.0033 | [−0.0131, +0.0066] | No |
| Accuracy, matching | −0.0122 | [−0.0490, +0.0259] | No |
| Accuracy, MCQ | −0.0220 | [−0.0522, +0.0073] | No |
| Accuracy, MCQ no-image | −0.0139 | [−0.0424, +0.0171] | No |

Passage/image contribution: 0.183 (Qwen2.5-VL-7B), 0.199 (Qwen3-VL-4B).


## Verifier screen

Two independent Gemini screeners checked whether any candidate answer is
also a correct answer to a *different* question in the same pool (the
paper's Section 3.3 well-formedness check, extended here to attach each
question's own image so the screener judges with the same information the
evaluated model had).

| Screener config | Groups flagged | Retention |
|---|---|---|
| 1 screener, any objection | 20/256 | 92.2% |
| 2 screeners, either objects (OR) | 104/256 | 59.4% |
| **2 screeners, both must agree (AND)** | **11/256** | **95.7%** |

The 2-of-2 result is the intended, reported check — closest to the paper's
own 2-of-3 majority design, adjusted for only 2 screeners being reachable
via a standard (non-Vertex) API key.

---

## Interpretation

**Accuracy is statistically blind to confabulation — reproduced on a new
benchmark and modality.** The two models are statistically indistinguishable
on every accuracy measure (all CIs include zero), yet differ sharply and
significantly in false-answer rate: Qwen3-VL-4B fabricates an answer 98.8%
of the time when the correct one has been removed from the pool, versus
94.7–94.9% for Qwen2.5-VL-7B. **A leaderboard based on accuracy alone would
call these two models equal. They are not equal in how much their answers
can be trusted.** This is the paper's central finding (Section 4.2),
reproduced here on a vision-language benchmark neither the original paper
nor, to our knowledge, any other published work has tested it against.

**De-saturation transfers to the visual setting.** Matching collapses MCQ
accuracy from ~46–48% to ~5–7% exact assignment, with the guessing floor
dropping several orders of magnitude — the same qualitative pattern as the
paper's five text-only benchmarks.

**Image dependency is real but leaves a residual leak.** Removing the image
drops accuracy from 46.3–48.5% (MCQ with image) to 34.3–35.7% (text only) —
a genuine reliance on vision — but 34–36% remains well above the 25% chance
floor for 4-option MCQ, indicating a fraction of MMMU-Pro's standard-4-option
questions are answerable from text alone. This is the specific leak
MMMU-Pro's own construction was designed to filter out; it is not fully
eliminated in this configuration.

## Limitations

1. **Two models, one seed.** No repeated runs; cross-architecture float
   variance was not characterized beyond a single spot-check.
2. **One corrupted image** (of 1,342, 0.07%) was skipped with a fallback
   rather than repaired; negligible at this rate but worth noting.
3. **The verifier screen uses 2 of the paper's 3 intended screeners** — the
   third (`gemma-4-26b-a4b-it-maas`) requires enterprise Vertex access not
   available via a standard API key.

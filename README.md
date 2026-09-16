# Answer Pooling — Extended to Vision-Language Benchmarks

Extends the [Answer Pooling](https://github.com/AbobakerAhmed/AnswerPool)
de-saturation method — pool the answer options of several questions into one
shared candidate list, withhold some golds to test for confabulation — from
its original five text-only benchmarks (QuALITY, RACE, MMLU-Pro, GPQA,
C-Eval) to **GPQA Diamond, HellaSwag, MMMU, MMMU-Pro, MathVista, and
ScienceQA**, four of which are vision-language benchmarks the original paper
never tested.

> Barebood, Ahmed, Bu Subait, Eltahir, Hussain. *"Your Benchmark Is Not
> Saturated: Reviving Multiple-Choice Evaluation with Answer Pooling."*
> KAUST Academy, 2026.

## What this adds over the original repo

- **`bench_datasets.py`** — loaders for the six new benchmarks, including
  full image handling (attachment, per-question labelling, token-budget
  estimation) for the four vision-language ones.
- **Multimodal support end-to-end** — `run_matching.py` attaches images on
  every backend (Gemini, Anthropic, OpenAI-compatible, and local vLLM).
- **A model registry (`models.py`)** — one flag switches between Claude,
  GPT, Gemini, Kimi, or local Qwen weights; per-provider quirks (key
  routing, reasoning-effort, base URLs) resolve automatically.
- **`report_tables.py` / `consolidate_report.py`** — regenerate the paper's
  Tables 3/4/7/8 layout from raw result files, independently cross-checked
  against each other.
- **Kaggle and Ibex-ready notebooks** (`notebooks/`) — including a
  known-issues table for the vLLM/transformers/CUDA-graph conflicts we hit
  running vision models on T4 GPUs.

## Results

**[MMMU-Pro: Qwen2.5-VL-7B-Instruct vs. Qwen3-VL-4B-Instruct](results/MMMU-Pro_AnswerPooling_Results.md)**
— reproduces the paper's central finding (accuracy is statistically blind to
confabulation) on a new modality: two models tied on every accuracy measure
differ significantly in false-answer rate (94.7–94.9% vs. 98.8%,
95% CI excludes zero).

The raw, executed notebook behind these numbers is in
[`notebooks/mmmu_pro_reproduction.ipynb`](notebooks/mmmu_pro_reproduction.ipynb) —
real cells, real outputs, including the environment fights and one known
inconsistency that's flagged rather than hidden.

## Quickstart

```bash
pip install -r requirements.txt

# does a benchmark's data fit the method? (no model needed)
python check_fit.py --dataset mmmu_pro --n 5 --min-options 4

# build the four arms
python build_matching.py --dataset mmmu_pro --n 5 --distractors \
    --min-options 4 --withhold 0.4 --out mmmu_pro_match.jsonl
python build_matching.py --dataset mmmu_pro --min-options 4 --arm mcq \
    --mirror mmmu_pro_match.jsonl --out mmmu_pro_mcq.jsonl

# run against a model (see models.py for the full registry)
python run_matching.py --in mmmu_pro_match.jsonl --model opus-5
python run_matching.py --in mmmu_pro_match.jsonl \
    --model Qwen/Qwen2.5-VL-7B-Instruct --backend vllm --tp 2

# tables
python report_tables.py --bench mmmu_pro --dir . --out report
```

No local GPU? See [`docs/KAGGLE.md`](docs/KAGGLE.md) (free T4s) or
[`docs/IBEX.md`](docs/IBEX.md) (KAUST HPC, A100s).

## Which benchmarks actually fit the method

Not every benchmark suits answer pooling equally. See
[`docs/BENCHMARK_FIT.md`](docs/BENCHMARK_FIT.md) for the full analysis;
summary:

| Benchmark | Fit | Why |
|---|---|---|
| GPQA Diamond | Good | Adversarial expert distractors, clean grouping |
| ScienceQA | Good | Large topic groups, clean once filtered to ≥4 options |
| MMMU-Pro | Good | 95.7% verifier retention — cleanest tested so far |
| MMMU | Partial | Many items answerable without the image |
| HellaSwag | Partial | Machine-generated distractors leak style cues |
| MathVista | Weak | Half free-form, numeric-answer collisions, non-adversarial distractors |

## Repository layout

```
build_matching.py       pool/withhold/mirror — the core transform
run_matching.py          model inference, every backend, multimodal
bench_datasets.py        loaders for the 6 new benchmarks
models.py                model registry (Claude/GPT/Gemini/Kimi/local)
check_fit.py             does a benchmark's data satisfy the method?
report_tables.py         paper-format Tables 3/4/7/8
consolidate_report.py    independent cross-check of report_tables.py
verify_wellformed.py     model-based ambiguity screen (paper Section 3.3)
smoke_test.py            offline end-to-end test, no keys/GPU needed
notebooks/               Kaggle + Ibex templates, and the real MMMU-Pro run
docs/                    platform setup, troubleshooting, benchmark fit
results/                 written-up findings
```

## Verify the pipeline works before spending API/GPU budget

```bash
python smoke_test.py
```

Builds and scores all six new benchmark configurations against synthetic
data — no network, no API keys, no GPU. If this passes, the only things
that can go wrong on a real run are credentials or dataset access.

## Citing

If you use this extension, please cite the original paper and note this
repo as an unofficial extension:

```bibtex
@misc{answerpool2026,
  title={Your Benchmark Is Not Saturated: Reviving Multiple-Choice Evaluation with Answer Pooling},
  author={Barebood, Nawaf and Ahmed, Abobaker and Bu Subait, Hussain and Eltahir, Mohamed and Hussain, Tanveer},
  year={2026},
  howpublished={KAUST Academy}
}
```

## License

See [`LICENSE`](LICENSE) — currently a placeholder pending verification
against the original repository's licensing terms.

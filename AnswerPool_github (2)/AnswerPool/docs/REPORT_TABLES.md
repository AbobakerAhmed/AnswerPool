# Reproducing the paper's report tables

## Reproducing the report's tables on the new benchmarks

`run_report.sh` runs the paper's full matrix and `report_tables.py` renders
Tables 3, 4, 7 and 8 in the report's layout, as Markdown and booktabs LaTeX:

```bash
MODELS="opus-5 gpt-5.2 kimi-k3 gemini-2.5-pro" ./run_report.sh
python report_tables.py --all --dir runs --out runs/report_tables
```

Per benchmark it builds five item-paired arms: `match` (withhold p=0.4, Table
4), `match0` (no withhold, mirrored onto the same groups, Table 3's exact
assignment), `mcq` (mirror-paired), `co` (choices-only, alpha) and, on the
visual benchmarks, `noimg` (image-withheld MCQ, the closed-book control). The
tables report per-question and exact-assignment accuracy, group-scored MCQ,
spread, Spearman rho, guessing floors, false-answer / false-abstention /
strict rates, paired matching-minus-MCQ contrasts with group-resampled 95% CIs,
and alpha with a bootstrap test against 1/M. `unparsed` is a column because
API backends have no grammar-constrained decoding (Section 3.5).

Benchmark recipes in the driver follow the fit analysis above: MMMU-Pro
standard config (never `vision`), ScienceQA and MathVista at >=4 options,
MathVista grouped by source, MMMU-Pro 10-option at N=3 keeping 7 of 9.


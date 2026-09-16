# Does each benchmark fit the Answer Pooling method?

## Does each benchmark fit the method?

The transform needs (Sections 3.1 to 3.5 of the paper): several questions per
document or topic, authored distractors per question that the pool retains,
unique golds within a group (W1), each candidate defensible for at most one
question (W2), a pool of at most 26 letters, and, wherever a context exists,
the closed-book filter. `python check_fit.py --dataset <name>` measures the
data-side properties; the model-side screens are `--arm choices_only`
(separability alpha), `verify_wellformed.py` (W2) and `--arm mcq_no_passage`
+ `filter_blind.py` (closed-book, images withheld on visual benchmarks).

| benchmark | fit | flags |
|---|---|---|
| GPQA Diamond | yes | only 198 items, ~30 groups at N=5: wide CIs, use N=3 or gpqa_main |
| HellaSwag | partial | machine-written distractors (expect high alpha, as on MMLU-Pro); generic endings threaten W2, run the verifier screen |
| MMMU | partial | the image is the passage: many items are answerable without it, so the closed-book filter is required; 900 labelled items; numeric options collide (W1) |
| MMMU-Pro standard | partial | 10-option config must keep 7 of 9 distractors at N=3 (`mmmu_pro10` in the driver), see Section 4.5 of the paper |
| MMMU-Pro vision | **no** | each image prints the question's own options, so candidate ownership leaks and pooling collapses to per-item MCQ |
| MathVista | poor | 46% free-form dropped; numeric golds lose most groups to W1; distractors not adversarially authored; task-level pools mix answer types (easy-matching regime) |
| ScienceQA | yes | run the image-withheld filter on the IMG subset; templated and yes/no answers cause W1 skips |

Visual benchmarks need a vision model on every backend; the paper's local
models (Qwen3, Gemma 3 text) cannot run them.


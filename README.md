# Answer Pooling

Make a saturated multiple-choice benchmark hard again, using only the labels it
already ships and find out whether the model admits when it cannot answer.

Pool the options of *N* questions that share a document into one exclusive
candidate list and ask the model to assign all *N* at once. Nothing is authored
and no label is edited, so the transform inherits whatever validity the source
benchmark had. Delete a gold from the pool and that question becomes
unanswerable, so confabulation is scored in the same forward pass as accuracy,
with no judge.

Two things change, and neither is a new label. Exclusivity couples the items,
so one wrong pick costs twice. And every candidate is a plausible answer to
*some* question in the pool, which removes the option-elimination shortcut that
keeps MCQ scores inflated.

On QuALITY the best evaluated model falls from 0.91 (MCQ) to 0.52 (exact
assignment) while the guessing floor drops from 0.25 to 5.4e-07, and model
rankings are preserved.

---

## Install

```bash
pip install -r requirements.txt
```

`vllm` is optional and only needed for local open-weight models. API models go
through Vertex: set `VERTEX_KEY=/path/to/key.json` or `GOOGLE_API_KEY=...`.

Gated datasets (GPQA) and gated models need `HF_TOKEN`. To keep weights and
datasets off your home quota, set `HF_HOME=/path/to/cache`.

## Quickstart

Convert QuALITY, run a model, score it:

```bash
python build_matching.py --n 5 --distractors --out groups.jsonl
python run_matching.py --in groups.jsonl --model Qwen/Qwen3-8B --backend vllm --limit 300
python compare.py --glob "groups.*.results.jsonl"
```

Add the abstention axis by withholding 40 percent of gold answers:

```bash
python build_matching.py --n 5 --distractors --withhold 0.4 --out wh40.jsonl
python run_matching.py --in wh40.jsonl --model Qwen/Qwen3-8B --backend vllm --limit 300
```

`compare.py` then reports accuracy, exact assignment, false-answer and
false-abstention rates, and paired bootstraps.

## What gets measured

| metric | floor | what it catches |
|---|---|---|
| per-pair accuracy | 1/M | partial credit, keeps statistical power |
| exact assignment | 1/(M!/(M-N)!) | the headline number, every slot right |
| false-answer rate | — | answered a question whose gold was withheld |
| false-abstention rate | — | declined when the gold was in fact present |
| parse failure | — | reported separately, never folded into wrong answers |

Both floors are printed by the builder for the *N* and *M* you actually built.

## Supported benchmarks

| `--dataset` | context | grouped by | notes |
|---|---|---|---|
| `quality` (default) | long passage | document | 13 to 20 questions per article |
| `race` | short passage | document | `--race-config all\|high\|middle` |
| `mmlupro` | none | topic | 10 options, use `--max-distractors 7 --n 3` |
| `gpqa` | none | subdomain | gated on the Hub, needs `HF_TOKEN` |
| `ceval` | none | subject | Chinese |

Passage-free benchmarks cannot run the arms that manipulate passage
availability (`no_passage`, `freeform`, `mcq_none`, `mcq_prose`).

## Arms

| `--arm` | what the model sees |
|---|---|
| `matching` | the pooled assignment task (the transform) |
| `mcq` | standard MCQ, one question at a time, the paired baseline |
| `no_passage` / `mcq_no_passage` | the same without the passage (closed-book controls) |
| `choices_only` | the pool alone, no questions and no passage (partial-input probe) |
| `mcq_choices_only` | options alone, per item, the uncapped version of the same probe |
| `mcq_none` | delete-gold MCQ with "none of these" printed as an option |
| `mcq_prose` | delete-gold MCQ with a prose permission to write none, no printed option |
| `easy` | pooled, but padded from unrelated documents (format-tax control) |
| `freeform` | open-ended, mismatched passage, judged by `judge_freeform.py` |

**Arms that must be mirror-built.** `mcq`, `mcq_no_passage`, `mcq_none`,
`mcq_prose`, `easy` and `freeform` take `--mirror <a built matching file>` so
they inherit its exact questions and withheld set. Do not build them
independently: group ids match but the questions inside them will not, because
per-group rng consumption differs between arms.

```bash
python build_matching.py --arm mcq --mirror groups.jsonl --out groups_mcq.jsonl
```

## The screens

Three judge-free filters for questions a benchmark should not be scoring.

**Solvable without the context.** Run the closed-book arm, then filter:

```bash
python build_matching.py --arm mcq_no_passage --out np.jsonl
# run np.jsonl on 3 models, then
python filter_blind.py --glob "np.*.results.jsonl" --votes 2 --out blind.txt
python build_matching.py --n 5 --distractors --exclude blind.txt --out filtered.jsonl
```

**Ambiguous once pooled.** Screener models see the gold revealed and judge
whether any other candidate also answers:

```bash
python verify_wellformed.py --src groups.jsonl --out flagged.txt
python compare.py --glob "groups.*.results.jsonl" --exclude-groups flagged.txt
```

**Separable by style.** Run `--arm choices_only`. Accuracy at the 1/M floor
means a style-homogeneous pool. Anything above it, bounded by 1/N, measures how
far golds stand apart from their distractors.

## File naming

Both stages write JSONL. `run_matching.py` derives its output name from the
input file and the model, and `compare.py` reads that convention to label rows:

```
<stem>.jsonl  ->  <stem>.<model>.results.jsonl
```

Keep that shape if you rename anything, or the comparison tables will merge or
mislabel runs.

## Analysis scripts

| script | question it answers |
|---|---|
| `compare.py` | all tables, paired bootstraps, rankings |
| `verify_wellformed.py` | does pooling leave the questions uniquely answerable? |
| `filter_blind.py` | which questions are answerable without the context? |
| `validity.py` | which closed-form measure tracks open-ended confabulation? |
| `own_set.py` | do errors come from a question's own options or its neighbours'? |
| `position_bias.py` | does the printed none option's position affect abstention? |
| `judge_freeform.py` | classify open-ended responses as ANSWER or ABSTAIN |
| `rescore.py` | re-parse saved raw outputs offline, no re-runs |
| `estimate_filter.py`, `diagnose.py`, `check_data.py` | dataset and scoring sanity checks |

Run any of them with `--help` for the full flag list.

## Notes that will save you time

**Constrained decoding is on by default for vLLM.** Output format compliance
becomes 100 percent by construction. `--no-guided` disables it and is only
useful as a control. Without it, format failures masquerade as wrong answers.

**Resume is content-checked.** Every result stores a hash of its exact prompt,
so re-running after rebuilding an input discards stale answers rather than
silently pairing old responses with new questions.

**Raw outputs are stored** (4000 chars), so `rescore.py` can re-parse
everything offline if the parser improves. A shorter cap once truncated the
outputs of a model that echoes questions before answering, which invalidated a
run.

**The answer alphabet ends at Z.** With `--distractors`, `N > 6` overflows a
26-letter pool. The builder skips such groups and reports the count.

**Verify alignment after any rebuild:**

```bash
python -c "
import json
def q(f, per_q):
    d = {}
    for l in open(f, encoding='utf-8'):
        l = l.strip()
        if not l: continue
        r = json.loads(l)
        if per_q: d[(r['group_id'], r.get('q_index', 0))] = r['questions'][0]
        else:
            for i, x in enumerate(r['questions']): d[(r['group_id'], i)] = x
    return d
a, b = q('groups.jsonl', False), q('groups_mcq.jsonl', True)
sh = set(a) & set(b)
print('identical:', sum(a[k] == b[k] for k in sh), '/', len(sh))
"
```

## References

The transform authors nothing, so every question and label comes from the
source benchmarks below. Cite them alongside this repo if you publish numbers.

- **QuALITY** — Pang et al., *QuALITY: Question Answering with Long Input
  Texts, Yes!*, NAACL 2022. [arXiv:2112.08608](https://arxiv.org/abs/2112.08608)
- **RACE** — Lai et al., *RACE: Large-scale ReAding Comprehension Dataset From
  Examinations*, EMNLP 2017. [arXiv:1704.04683](https://arxiv.org/abs/1704.04683)
- **MMLU-Pro** — Wang et al., *MMLU-Pro: A More Robust and Challenging
  Multi-Task Language Understanding Benchmark*, NeurIPS 2024 Datasets and
  Benchmarks. [arXiv:2406.01574](https://arxiv.org/abs/2406.01574)
- **GPQA** — Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*,
  COLM 2024. [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)
- **C-Eval** — Huang et al., *C-Eval: A Multi-Level Multi-Discipline Chinese
  Evaluation Suite for Foundation Models*, NeurIPS 2023.
  [arXiv:2305.08322](https://arxiv.org/abs/2305.08322)

Local inference uses vLLM — Kwon et al., *Efficient Memory Management for Large
Language Model Serving with PagedAttention*, SOSP 2023.
[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

## Citation

Paper in preparation. Until it is public, cite the repository:

```bibtex
@software{answer_pooling,
  title  = {Answer Pooling: turning saturated multiple-choice benchmarks
            into exclusive assignment tasks},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/TODO/answer-pooling}
}
```

## License

MIT.

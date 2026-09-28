# Pooled benchmark manifests

One gzipped JSONL file per build the paper reports, named
`<stem>.manifest.jsonl.gz` and written by `python -m AnswerPooling.manifest export`.
A manifest holds no question, option, or passage text.

**First line, the header:**

| Field | Meaning |
|---|---|
| `stem` | build name, e.g. `h_match` (QuALITY, pooled) or `ra_mcq` (RACE, multiple choice) |
| `dataset` | `--dataset` value for `build_matching` |
| `hub_dataset` | Hugging Face dataset the build was made from |
| `hub_revision` | dataset revision, for `--revision` |
| `records` | number of groups, or items for multiple-choice builds |
| `templates` | prompt templates by id: the prompt with every passage, question, and option replaced by a slot `⦃P0⦄`, `⦃Q0⦄`, `⦃C0⦄` |

**Every other line, one group or item:**

| Field | Meaning |
|---|---|
| `group_id` | group identifier, shared by every arm built on the same questions |
| `arm` | `matching` (pooled), `mcq`, `no_passage`, `mcq_none`, ... |
| `n`, `m` | questions per group and options in the pool |
| `t` | id of the prompt template |
| `p`, `q`, `c` | identifiers of the passages, questions, and options that fill the template's slots, in slot order |
| `qp`, `qm` | cross-domain passage labels and image citations of each question, as they appear in the prompt |
| `answer` | correct letter per question, `none` for a removed answer |
| `withheld` | positions of the questions whose answer was removed |
| `phash` | hash of the exact prompt |
| `img` | image files, in the order they are attached |
| `item_id`, `q_index` | multiple-choice items: the source group and position |
| `source_groups`, `cross_domain` | cross-domain groups: the groups their questions came from |

Identifiers are the first 16 hex digits of the SHA-1 of the normalised text
(whitespace collapsed; for questions, `<image k>` citations canonicalised and
the cross-domain `[Passage k]` label removed).

**Screen outputs:**

- `blind_solvable.txt`: hashes of the QuALITY questions removed by the
  no-context filter.
- `flagged_groups.txt`: group ids with a question flagged by the ambiguity check.

**Rebuilding.** Make one universe build per benchmark (`build_matching --n 1`,
see the main README), then:

```bash
python -m AnswerPooling.manifest rebuild --universe "universe/*.jsonl" --out-dir rebuilt
python -m AnswerPooling.manifest verify --dir rebuilt
```

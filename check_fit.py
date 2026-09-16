#!/usr/bin/env python
# ============================================================================
# check_fit.py — does a benchmark satisfy what the answer-pooling transform
# needs? Measures, on the real data, the properties that decide it:
#
#   groups         how many N-groups the topic structure yields (Table 2 has
#                  83 to 400 per benchmark; below ~50 the CIs get wide)
#   dup-gold loss  fraction of groups the W1 screen (unique golds) rejects.
#                  Numeric-answer benchmarks lose most of their groups here
#   option types   share of golds that are numbers / yes-no / single words.
#                  A pool that mixes types is separable by type, which is the
#                  easy-matching regime (Section 3.3): no difficulty is gained
#   options/item   pool size M = sum of options; must stay <= 26 letters
#   candidate reuse how often the SAME option string appears in several
#                  questions of a topic (W2 pressure: a candidate defensible
#                  for more than one question)
#
# The verifier screen (W2 semantic ambiguity), the choices-only arm (gold vs
# distractor separability, alpha) and the closed-book filter are MODEL runs
# and cannot be read off the data; run verify_wellformed.py, --arm
# choices_only and --arm mcq_no_passage for those.
#
#   python check_fit.py --dataset scienceqa
#   python check_fit.py --dataset mathvista --n 3
# ============================================================================
import argparse
import random
import re
from collections import Counter, defaultdict

from bench_datasets import load_rows, norm, VISUAL

NUM = re.compile(r"^[-+]?\$?\d[\d,]*(\.\d+)?\s*[%a-zA-Z°]{0,4}$")
YESNO = {"yes", "no", "true", "false"}


def kind(s):
    t = norm(s).lower().rstrip(".")
    if t in YESNO:
        return "yes/no"
    if NUM.match(t):
        return "number"
    if len(t.split()) == 1:
        return "word"
    if len(t.split()) <= 4:
        return "short phrase"
    return "sentence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["gpqa_diamond", "hellaswag", "mmmu", "mmmu_pro",
                             "mathvista", "scienceqa"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-distractors", type=int, default=3)
    ap.add_argument("--min-options", type=int, default=0)
    ap.add_argument("--splits", default="")
    ap.add_argument("--mmmu-pro-config", default="standard (4 options)")
    ap.add_argument("--group-by-visual", default="")
    ap.add_argument("--require-image", action="store_true")
    a = ap.parse_args()
    a.gpqa_config = "gpqa_diamond"
    a.mmmu_split = "validation"
    a.subjects = ""
    a.mathvista_split = "testmini"
    if not a.group_by_visual:
        a.group_by_visual = "task" if a.dataset == "mathvista" else "topic"
    if not a.splits:
        a.splits = "validation" if a.dataset == "hellaswag" else "test"

    rows = load_rows(a.dataset, random.Random(0), a)
    if a.min_options:
        before = len(rows)
        rows = [r for r in rows if len(r["options"]) >= a.min_options]
        print(f"--min-options {a.min_options}: {before-len(rows)} dropped, "
              f"{len(rows)} remain")
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["_topic"]].append(r)

    n_opts = Counter(len(r["options"]) for r in rows)
    gold_kind = Counter(kind(r["options"][r["answer"]]) for r in rows)
    # groups: chunk each topic like the builder does; W1 loss with a shuffle
    rng = random.Random(0)
    total = kept = over26 = 0
    reuse = 0
    for items in by_topic.values():
        rng.shuffle(items)
        seen = Counter(norm(o) for r in items for o in r["options"])
        reuse += sum(1 for c in seen.values() if c > 1)
        for st in range(0, len(items) - a.n + 1, a.n):
            chunk = items[st:st + a.n]
            total += 1
            golds = [norm(r["options"][r["answer"]]) for r in chunk]
            if len(set(golds)) != len(golds):
                continue
            m = len({norm(o) for r in chunk for o in
                     ([r["options"][r["answer"]]]
                      + [o for i, o in enumerate(r["options"]) if i != r["answer"]][:a.max_distractors])})
            if m > 26:
                over26 += 1
                continue
            kept += 1
    leftover = sum(len(v) % a.n for v in by_topic.values())

    print(f"\n=== {a.dataset}  ({len(rows)} usable items, {len(by_topic)} topics, N={a.n}) ===")
    print(f"topics with < N items (unusable): "
          f"{sum(1 for v in by_topic.values() if len(v) < a.n)} / {len(by_topic)}; "
          f"{leftover} items left over after chunking")
    print(f"groups: {kept} kept of {total} candidate groups "
          f"(W1 duplicate golds rejected {total-kept-over26}, "
          f"pool > 26 letters rejected {over26})")
    print(f"options per item: {dict(sorted(n_opts.items()))}")
    print(f"gold answer types: " + ", ".join(f"{k} {v/len(rows):.0%}" for k, v in gold_kind.most_common()))
    print(f"option strings shared by >1 question within a topic: {reuse}")
    if a.dataset in VISUAL:
        n_img = sum(1 for r in rows if r["_images"])
        print(f"items with images: {n_img}/{len(rows)}  -> run --arm mcq_no_passage on "
              f"3 models and filter_blind.py before trusting matching numbers")

    flags = []
    if kept < 50:
        flags.append(f"FEW GROUPS ({kept}): statistical power will be low; try --n 3")
    if total and (total - kept - over26) / total > 0.3:
        flags.append("W1: >30% of groups lost to duplicate golds (numeric / templated answers)")
    if over26:
        flags.append("pool overflow: use --max-distractors so that N*(1+d) <= 26")
    top = gold_kind.most_common(1)[0][1] / len(rows)
    if top < 0.6 and len(gold_kind) >= 3:
        flags.append("MIXED OPTION TYPES: pool is separable by type (easy-matching regime); "
                     "group more finely (--group-by-visual source/skill) or restrict types")
    two = sum(v for k, v in n_opts.items() if k <= 2) / len(rows)
    if two > 0.15:
        flags.append(f"{two:.0%} of items have <=2 options: the MCQ baseline "
                     f"floor is 0.50 there and each adds one distractor to the "
                     f"pool. Use --min-options 3 (or 4) when building")
    if len(n_opts) > 1:
        flags.append("option count varies per item, so pool size M is not "
                     "constant: quote the worst-case floor or use --min-options")
    if gold_kind.get("yes/no", 0) / len(rows) > 0.1:
        flags.append("many yes/no items: 2-option items shrink M and always duplicate golds")
    if a.dataset == "hellaswag":
        flags.append("machine-written distractors: expect a high choices-only alpha "
                     "(Balepur artifact); W2 (generic endings) needs verify_wellformed.py")
    if a.dataset == "mmmu_pro" and a.mmmu_pro_config == "vision":
        flags.append("NOT POOLABLE: the vision config prints each question's own options "
                     "inside its image, so candidate ownership leaks")
    if a.dataset == "mathvista":
        flags.append("46% free-form items dropped; distractors are source-generated, "
                     "not adversarially authored: little difficulty to recover")
    print("\nflags:" if flags else "\nflags: none")
    for f in flags:
        print("  !", f)


if __name__ == "__main__":
    main()

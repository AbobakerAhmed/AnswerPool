#!/usr/bin/env python
# ============================================================================
# own_set.py — where does the extra difficulty of pooling actually come from?
#
# In the distractor-pooled arm the candidate list holds each question's OWN
# four options plus 4(N-1) options belonging to its neighbours. Every answer
# therefore lands in exactly one of three buckets:
#
#   own gold        the correct answer
#   own distractor  an error MCQ would also have allowed (same option set)
#   foreign         an error only the pooled format can produce
#
# That splits the matching-minus-MCQ gap into a within-question component and
# a cross-question component. Two derived numbers matter:
#
#   own-set rate          how often the model answers from the right question's
#                         own options at all, against a 4/M chance baseline.
#                         This is localisation ability, independent of whether
#                         the pick is correct.
#   conditional accuracy  accuracy GIVEN the model stayed inside the own set.
#                         If this matches the model's MCQ accuracy, pooling does
#                         not degrade within-question discrimination and the
#                         whole gap is cross-question confusion.
#
# On withhold arms the gold is absent, so a false answer splits into own
# distractor (an MCQ-style misjudgement) versus foreign (taking a candidate
# that belongs to another question).
#
# Runs entirely on existing files: results join to the source build by
# group_id, and per-question option sets come from the dataset by question text.
#
#   python -m AnswerPooling.own_set --src h_match.jsonl --glob "h_match.*.results.jsonl"
#   python -m AnswerPooling.own_set --src h_wh40.jsonl --glob "h_wh40.*.results.jsonl"
#   python -m AnswerPooling.own_set --src ra_match.jsonl --glob "ra_match.*.results.jsonl" --dataset race
# ============================================================================
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from .run_matching import label


def norm(s):
    return " ".join(str(s).split()).strip()


def load_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def question_options(dataset, splits, race_config="all"):
    """-> {normalised question text: [normalised options]}"""
    from datasets import load_dataset
    out = {}
    if dataset == "quality":
        for sp in splits.split(","):
            for r in load_dataset("emozilla/quality", split=sp.strip()):
                out.setdefault(norm(r["question"]), [norm(o) for o in r["options"]])
    elif dataset == "race":
        for sp in splits.split(","):
            for r in load_dataset("ehovy/race", race_config, split=sp.strip()):
                out.setdefault(norm(r["question"]), [norm(o) for o in r["options"]])
    elif dataset == "mmlupro":
        for r in load_dataset("TIGER-Lab/MMLU-Pro", split="test"):
            out.setdefault(norm(r["question"]), [norm(o) for o in r["options"]])
    elif dataset == "gpqa":
        for r in load_dataset("Idavidrein/gpqa", "gpqa_main", split="train"):
            out.setdefault(norm(r["Question"]),
                           [norm(r["Correct Answer"]), norm(r["Incorrect Answer 1"]),
                            norm(r["Incorrect Answer 2"]), norm(r["Incorrect Answer 3"])])
    elif dataset == "ceval":
        from datasets import get_dataset_config_names
        for subj in get_dataset_config_names("ceval/ceval-exam"):
            for r in load_dataset("ceval/ceval-exam", subj, split="val"):
                out.setdefault(norm(r["question"]),
                               [norm(r["A"]), norm(r["B"]), norm(r["C"]), norm(r["D"])])
    elif dataset == "mmmupro":
        # pooled questions carry renumbered <image k> markers, so key by the
        # builder's qkey(), which canonicalises the markers away
        import ast
        from .build_matching import qkey
        for r in load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test"):
            opts = r["options"]
            if isinstance(opts, str):
                opts = ast.literal_eval(opts)
            out.setdefault(qkey(norm(r["question"])), [norm(o) for o in opts])
    elif dataset == "medxpertqa":
        # the builder prepends "<image k>" citations and cuts the embedded
        # "Answer Choices:" block, so rebuild the same text here
        import json as _json
        from huggingface_hub import hf_hub_download
        from .build_matching import qkey
        jp = hf_hub_download("TsinghuaC3I/MedXpertQA", "MM/test.jsonl", repo_type="dataset")
        for line in open(jp, encoding="utf-8"):
            if not line.strip():
                continue
            r = _json.loads(line)
            q = re.split(r"\s*Answer Choices\s*:", norm(r["question"]), maxsplit=1)[0].strip()
            od = r["options"]
            if isinstance(od, str):
                od = _json.loads(od)
            n_img = len(r.get("images") or [])
            cites = " ".join(f"<image {k}>" for k in range(1, n_img + 1))
            out.setdefault(qkey(norm(f"{cites} {q}")), [norm(od[L]) for L in sorted(od)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the built .jsonl the results came from")
    ap.add_argument("--glob", default="", help="defaults to <src stem>.*.results.jsonl")
    ap.add_argument("--dataset", default="quality",
                    choices=["quality", "race", "mmlupro", "gpqa", "ceval", "mmmupro",
                             "medxpertqa"])
    ap.add_argument("--splits", default="validation,train")
    ap.add_argument("--race-config", default="all")
    a = ap.parse_args()
    from .build_matching import qkey

    src = load_jsonl(a.src)
    if not src:
        raise SystemExit(f"no records in {a.src}")
    stem = os.path.basename(a.src).replace(".jsonl", "")
    pattern = a.glob or f"{stem}.*.results.jsonl"

    qopts = question_options(a.dataset, a.splits, a.race_config)
    print(f"{len(src)} groups | {len(qopts)} questions with option sets")

    # group_id -> list over question index of the letters of its OWN options
    own = {}
    missing = 0
    for r in src:
        cands = [norm(c) for c in r.get("candidates", [])]
        pos = defaultdict(list)
        for i, c in enumerate(cands):
            pos[c].append(label(i))
        per_q = []
        for q in r.get("questions", []):
            # cross-domain groups of a passage benchmark prefix each question
            # with its passage label
            q = re.sub(r"^\[Passage \d+\]\s*", "", norm(q))
            opts = qopts.get(qkey(q) if a.dataset in ("mmmupro", "medxpertqa") else q)
            if opts is None:
                missing += 1
                per_q.append(None)
                continue
            letters = set()
            for o in opts:
                letters.update(pos.get(o, []))
            per_q.append(letters)
        own[r["group_id"]] = per_q
    if missing:
        print(f"NOTE: {missing} questions not found in the dataset, skipped")

    m_avg = sum(r.get("m", 0) for r in src) / len(src)
    k_avg = sum(len(v) for v in qopts.values()) / max(1, len(qopts))  # options per item
    print(f"mean pool size M = {m_avg:.1f}, own-set chance baseline = "
          f"{k_avg/m_avg:.3f}\n")

    hdr = (f"{'model':26s} {'own-set':>8s} {'gold':>8s} {'own-dis':>8s} "
           f"{'foreign':>8s} {'acc|own':>8s} {'n':>6s}")
    print(hdr); print("-" * len(hdr))
    for path in sorted(glob.glob(pattern)):
        recs = load_jsonl(path)
        if not recs:
            continue
        model = os.path.basename(path).replace(".judged.results.jsonl", "") \
                                      .replace(".results.jsonl", "").split(".")[-1]
        c_gold = c_ownd = c_for = tot = 0
        for r in recs:
            per_q = own.get(r["group_id"])
            if per_q is None:
                continue
            wh = set(r.get("withheld", []))
            for i, (g, p) in enumerate(zip(r["gold"], r["pred"])):
                if i >= len(per_q) or per_q[i] is None:
                    continue
                if p is None or p == "none":
                    continue          # unparsed or abstained: no candidate chosen
                tot += 1
                if i not in wh and p == g:
                    c_gold += 1
                elif p in per_q[i]:
                    c_ownd += 1
                else:
                    c_for += 1
        if not tot:
            continue
        own_set = (c_gold + c_ownd) / tot
        cond = c_gold / max(1, c_gold + c_ownd)
        print(f"{model:26s} {own_set:8.3f} {c_gold/tot:8.3f} {c_ownd/tot:8.3f} "
              f"{c_for/tot:8.3f} {cond:8.3f} {tot:6d}")

    print("\nown-set = answered from the question's own four options (chance "
          f"{4/m_avg:.3f})")
    print("acc|own = accuracy given the model stayed inside the own set. Compare "
          "to the\n          model's MCQ accuracy: equality means pooling costs "
          "nothing WITHIN a\n          question and the entire gap is "
          "cross-question confusion.")
    print("On withhold arms the gold is absent, so 'gold' is 0 by construction "
          "and the\nsplit of false answers into own-distractor versus foreign is "
          "the quantity of\ninterest.")


if __name__ == "__main__":
    main()

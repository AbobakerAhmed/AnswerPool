#!/usr/bin/env python
# ============================================================================
# filter_blind.py — find questions answerable WITHOUT the passage, and emit an
# exclusion list so groups can be rebuilt from context-requiring items only.
#
# Why filtering on the BLIND run is clean:
#   conditioning on no-passage performance is independent of the
#   passage-vs-no-passage comparison you then make. (Filtering on FULL-context
#   success would bias that comparison -- you would be selecting items where the
#   full condition got lucky, and every other arm regresses toward the mean.)
#
# Uses whatever mcq_no_passage results exist. An item is blind-solvable if at
# least --votes models answer it correctly without the passage.
#
#   python filter_blind.py --glob "g_mcq_np*.results.jsonl" --votes 2
#   -> writes blind_solvable.txt (question hashes), then:
#   python build_matching.py --n 5 --distractors --exclude blind_solvable.txt ...
# ============================================================================
import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict


def qhash(q):
    return hashlib.sha1(" ".join(str(q).split()).strip().lower().encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*mcq_np*.results.jsonl")
    ap.add_argument("--votes", type=int, default=2,
                    help="models that must answer it blind for it to count as solvable")
    ap.add_argument("--out", default="blind_solvable.txt")
    a = ap.parse_args()

    hits = defaultdict(int)          # qhash -> models correct blind
    seen = defaultdict(int)          # qhash -> models that attempted it
    models = set()
    for path in sorted(glob.glob(a.glob)):
        model = os.path.basename(path).replace(".results.jsonl", "").split(".")[-1]
        n = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("arm", "").endswith("no_passage"):
                continue
            q = r.get("questions", [None])[0] if "questions" in r else None
            if q is None:                       # results files may omit questions
                continue
            h = qhash(q)
            seen[h] += 1
            g, p = r["gold"][0], r["pred"][0]
            if p is not None and p == g:
                hits[h] += 1
            n += 1
        if n:
            models.add(model)
            print(f"{model:34s} {n:5d} blind items")

    if not seen:
        raise SystemExit(
            "no usable records. The results files must contain a `questions` field; "
            "if they don't, re-run run_matching.py after adding it, or filter using "
            "the source .jsonl instead.")

    solvable = sorted(h for h, c in hits.items() if c >= a.votes)
    total = len(seen)
    print(f"\nmodels: {len(models)} | distinct questions with blind data: {total}")
    for k in range(len(models) + 1):
        c = sum(1 for h in seen if hits[h] == k)
        print(f"  answered blind by {k}/{len(models)} models : {c:5d} ({100*c/total:5.1f}%)")
    print(f"\nblind-solvable (>= {a.votes} votes): {len(solvable)}/{total} "
          f"= {100*len(solvable)/total:.1f}%  -> excluded")
    print(f"remaining context-requiring: {total-len(solvable)} "
          f"({100*(total-len(solvable))/total:.1f}%)")

    with open(a.out, "w", encoding="utf-8") as f:
        for h in solvable:
            f.write(h + "\n")
    print(f"\n-> {a.out}")
    print("rebuild with:  python build_matching.py --n 5 --distractors "
          f"--exclude {a.out} --out g_d_filt.jsonl")


if __name__ == "__main__":
    main()

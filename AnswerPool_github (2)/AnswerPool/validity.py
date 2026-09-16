#!/usr/bin/env python
# ============================================================================
# validity.py — which closed-form abstention measure predicts open-ended
# confabulation: matching's false-match, or mcq_none's?
#
# Four model means cannot settle this (and on the current criterion three of the
# four models are statistically tied). Joining at the ITEM level gives 600
# paired observations per model instead of one.
#
# Join key is (group_id, q_index) -- the same question in all three arms.
#
# Phi (== Matthews correlation for binary) is used rather than raw agreement,
# because the base rates differ wildly across arms (matching false-matches ~60%
# of the time, freeform ~12%) and agreement rewards whichever measure happens to
# share the criterion's base rate rather than whichever tracks it.
#
# CAVEAT worth stating in the paper: the arms are unanswerable for DIFFERENT
# reasons -- freeform withholds the passage's relevance (mismatched document),
# matching and mcq_none withhold the answer from the options while the passage
# stays correct. A weak item-level correlation is therefore not damning on its
# own; the model-level comparison is the conceptually cleaner test.
#
#   python validity.py --free "h_free40.*.judged.results.jsonl" \
#                      --match h_wh40 --mcqnone h_mcqnone40
# ============================================================================
import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def model_of(path):
    b = os.path.basename(path)
    for suf in (".judged.results.jsonl", ".results.jsonl"):
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    return b.split(".")[-1]


def phi(pairs):
    """Matthews correlation for two binary vectors given as (x, y) pairs."""
    n11 = sum(1 for x, y in pairs if x and y)
    n10 = sum(1 for x, y in pairs if x and not y)
    n01 = sum(1 for x, y in pairs if not x and y)
    n00 = sum(1 for x, y in pairs if not x and not y)
    num = n11 * n00 - n10 * n01
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    return (num / den) if den else 0.0


def boot_diff(a_pairs, b_pairs, iters=4000, seed=0):
    """CI on phi(a) - phi(b), resampling the SAME item indices for both so the
    two measures stay paired on the item."""
    keys = sorted(set(a_pairs) & set(b_pairs))
    if len(keys) < 20:
        return None
    rng = random.Random(seed)
    obs = phi([a_pairs[k] for k in keys]) - phi([b_pairs[k] for k in keys])
    diffs = []
    for _ in range(iters):
        smp = [keys[rng.randrange(len(keys))] for _ in keys]
        diffs.append(phi([a_pairs[k] for k in smp]) - phi([b_pairs[k] for k in smp]))
    diffs.sort()
    return obs, diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))], len(keys)


def collect(pattern, kind):
    """-> {model: {(gid, qi): confabulated_bool}} over UNANSWERABLE items only."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        recs = load(path)
        if not recs:
            continue
        m = model_of(path)
        for r in recs:
            wh = set(r.get("withheld", []))
            if not wh:
                continue
            gid = r["group_id"]
            if kind == "group":                 # one record, N questions
                for i in wh:
                    p = r["pred"][i] if i < len(r["pred"]) else None
                    if p is not None:
                        out[m][(gid, i)] = (p != "none")
            else:                               # one record, one question
                p = r["pred"][0] if r.get("pred") else None
                if p is None:
                    continue
                # q_index is NOT stored in results files -- defaulting it to 0
                # silently collapsed every item of a group onto (gid, 0), so the
                # join only hit groups whose question 0 was withheld (~40%) and
                # reported n=113 instead of 600. Recover it from the key, which
                # is the item_id "{group_id}_q{i}".
                qi = r.get("q_index")
                if qi is None:
                    mt = re.search(r"_q(\d+)$", str(r.get("key", "")))
                    if not mt:
                        continue
                    qi = int(mt.group(1))
                out[m][(gid, qi)] = (p != "none")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default="h_free40.*.judged.results.jsonl")
    ap.add_argument("--match", default="h_wh40")
    ap.add_argument("--mcqnone", default="h_mcqnone40")
    a = ap.parse_args()

    free = collect(a.free, "item")
    match = collect(f"{a.match}.*.results.jsonl", "group")
    mcqn = collect(f"{a.mcqnone}.*.results.jsonl", "item")

    print(f"{'model':26s} {'n':>5s} {'phi(match)':>11s} {'phi(mcq_none)':>14s} "
          f"{'difference':>26s}")
    print("-" * 88)
    pooled_a, pooled_b = {}, {}
    for m in sorted(set(free) & set(match) & set(mcqn)):
        keys = sorted(set(free[m]) & set(match[m]) & set(mcqn[m]))
        if len(keys) < 20:
            print(f"{m:26s} {len(keys):5d}   too few shared items")
            continue
        pa = {k: (match[m][k], free[m][k]) for k in keys}
        pb = {k: (mcqn[m][k], free[m][k]) for k in keys}
        r = boot_diff(pa, pb)
        d = f"{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]" if r else "-"
        sig = "  SIG" if r and not (r[1] <= 0 <= r[2]) else ""
        print(f"{m:26s} {len(keys):5d} {phi(list(pa.values())):11.3f} "
              f"{phi(list(pb.values())):14.3f} {d:>26s}{sig}")
        for k in keys:
            pooled_a[(m, k)] = pa[k]
            pooled_b[(m, k)] = pb[k]

    if pooled_a:
        r = boot_diff(pooled_a, pooled_b)
        print("-" * 88)
        print(f"{'POOLED':26s} {len(pooled_a):5d} {phi(list(pooled_a.values())):11.3f} "
              f"{phi(list(pooled_b.values())):14.3f}", end="")
        if r:
            sig = "  SIG" if not (r[1] <= 0 <= r[2]) else ""
            print(f" {f'{r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]':>26s}{sig}")
        else:
            print()
        print("\npositive difference = matching's false-match tracks open-ended "
              "confabulation\nbetter than mcq_none's does, at the item level.")


if __name__ == "__main__":
    main()

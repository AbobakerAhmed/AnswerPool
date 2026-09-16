#!/usr/bin/env python
# ============================================================================
# estimate_filter.py — what WOULD blind-filtering do? Answered from results
# already collected, no new API calls.
#
# mcq_no_passage records carry group_id + a per-question key, so they join to
# the matching groups. For each group we count how many of its N questions are
# blind-solvable, then stratify matching accuracy by that count.
#
# If matching accuracy is FLAT across strata, blind-solvable items are not what
# is inflating it -- the compatibility shortcut is structural to the format and
# filtering will not help. If it rises steeply with the count, filtering will.
#
#   python estimate_filter.py --blind "groups_n5_mcq_np.*.results.jsonl" \
#                             --match "g_d.*.results.jsonl"
# ============================================================================
import argparse
import glob
import json
import os
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
    return os.path.basename(path).replace(".results.jsonl", "").split(".")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", default="groups_n5_mcq_np.*.results.jsonl")
    ap.add_argument("--match", default="g_d.*.results.jsonl")
    ap.add_argument("--votes", type=int, default=2,
                    help="models that must answer blind for an item to count solvable")
    a = ap.parse_args()

    # ---- blind-solvability per (group_id, q_index) ------------------------
    hits, seen = defaultdict(int), defaultdict(int)
    models = set()
    for p in sorted(glob.glob(a.blind)):
        m = model_of(p)
        n = 0
        for r in load(p):
            if not r.get("arm", "").endswith("no_passage"):
                continue
            key = r.get("key", "")
            mt = re.search(r"_q(\d+)$", key)
            if not mt:
                continue
            qi = int(mt.group(1))
            k = (r["group_id"], qi)
            seen[k] += 1
            g, pr = r["gold"][0], r["pred"][0]
            if pr is not None and pr == g:
                hits[k] += 1
            n += 1
        if n:
            models.add(m); print(f"blind  {m:32s} {n:5d} items")
    if not seen:
        raise SystemExit("no blind records matched --blind")

    solvable = {k for k, c in hits.items() if c >= a.votes}
    print(f"\n{len(models)} models | {len(seen)} items with blind data")
    print(f"blind-solvable (>={a.votes} votes): {len(solvable)}/{len(seen)} "
          f"= {100*len(solvable)/len(seen):.1f}%")

    n_solv = defaultdict(int)
    for (gid, qi) in solvable:
        n_solv[gid] += 1

    # ---- matching accuracy stratified by that count ------------------------
    for p in sorted(glob.glob(a.match)):
        m = model_of(p)
        strata = defaultdict(lambda: [0, 0])          # count -> [ok, tot]
        for r in load(p):
            gid = r["group_id"]
            if gid not in {g for g, _ in seen}:
                continue
            wh = set(r.get("withheld", []))
            ok = tot = 0
            for i, (g, pr) in enumerate(zip(r["gold"], r["pred"])):
                if i in wh or pr is None:
                    continue
                tot += 1; ok += (pr == g)
            if tot:
                s = strata[n_solv.get(gid, 0)]
                s[0] += ok; s[1] += tot
        if not strata:
            continue
        print(f"\nmatching accuracy by #blind-solvable questions in the group — {m}")
        print(f"  {'#solvable':>10s} {'groups':>8s} {'acc':>8s}")
        tot_ok = tot_n = clean_ok = clean_n = 0
        for c in sorted(strata):
            ok, n = strata[c]
            print(f"  {c:>10d} {n//5 if n>=5 else 1:>8d} {ok/n:>8.4f}")
            tot_ok += ok; tot_n += n
            if c == 0:
                clean_ok += ok; clean_n += n
        if tot_n:
            print(f"  {'ALL':>10s} {tot_n//5:>8d} {tot_ok/tot_n:>8.4f}")
        if clean_n:
            print(f"  -> groups with ZERO blind-solvable questions: {clean_ok/clean_n:.4f} "
                  f"(n={clean_n} slots)")
            delta = clean_ok/clean_n - tot_ok/tot_n
            print(f"  -> estimated effect of filtering: {delta:+.4f}")
            if abs(delta) < 0.02:
                print("     FLAT -> blind-solvable items are NOT what inflates matching; "
                      "the shortcut is structural to the format and filtering won't fix it.")
            else:
                print("     filtering moves the number -- worth running for real.")


if __name__ == "__main__":
    main()

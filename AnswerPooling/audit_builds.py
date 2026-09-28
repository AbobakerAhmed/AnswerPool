#!/usr/bin/env python
# ============================================================================
# audit_builds.py -- two checks that every "identical items" claim rests on.
#
#   1. ALIGNMENT. Every arm of a benchmark must hold the same questions under
#      the same group id as the pooled build it is compared with. Arms that
#      were regenerated from the dataset instead of mirror-built can silently
#      hold different questions under the same id (the chunking is reshuffled
#      whenever any flag that feeds it differs). Full-pool arms are checked
#      against <prefix>_match.jsonl, removed-mode arms against the pooled
#      removed build (<prefix>_wh*.jsonl).
#   2. STALE RESULTS. Every results file must have been produced on the build
#      file that sits next to it now: the prompt hash stored with each answer
#      must equal the hash of the current prompt. A stale answer is paired by
#      group id with answers to a different prompt.
#
# Zero GPU, seconds. Prints problems only, plus one summary line per prefix.
#   python -m AnswerPooling.audit_builds --prefix h ra gp ce mp9 mm mx vm
# ============================================================================
import argparse
import glob
import json
import os
import re
from collections import defaultdict

from .run_matching import phash

IMG = re.compile(r"<\s*image\s*\d+\s*>", re.I)


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def qnorm(q):
    # single-question arms restore each question's own image numbering, the
    # pooled arm renumbers across the group: compare the text without markers
    return " ".join(IMG.sub(" ", str(q)).split())


def is_build(recs):
    return bool(recs) and "prompt" in recs[0] and "group_id" in recs[0]


def removed_family(recs):
    return any(r.get("withheld") for r in recs[:200]) or \
        recs[0].get("arm") in ("mcq_none", "mcq_prose")


def view(recs):
    """group id -> (question set, pool tuple or None for single-question arms)"""
    qs, pool = defaultdict(set), {}
    for r in recs:
        g = r["group_id"]
        qs[g].update(qnorm(q) for q in (r.get("questions") or []))
        if r.get("n", 1) > 1:
            pool[g] = tuple(r.get("candidates") or [])
    return {g: (frozenset(q), pool.get(g)) for g, q in qs.items()}


def audit(prefix, d):
    builds = {}
    for p in sorted(glob.glob(os.path.join(d, f"{prefix}_*.jsonl"))):
        base = os.path.basename(p)
        if base.endswith(".results.jsonl") or ".judged." in base:
            continue
        recs = load(p)
        if is_build(recs) and not recs[0].get("cross_domain") \
                and recs[0].get("arm") not in ("freeform", "choices_only"):
            builds[base[:-len(".jsonl")]] = recs

    print(f"\n=== {prefix}: {len(builds)} build files")
    ref_full = f"{prefix}_match" if f"{prefix}_match" in builds else None
    whs = sorted(s for s in builds if re.fullmatch(rf"{prefix}_wh\d+", s))
    views = {s: view(r) for s, r in builds.items()}
    n_bad_align = n_skip = 0
    # Only the arms the paper pairs with a pooled build are checked. The N
    # sweep, second seed, reuse and removal-fraction builds are separate
    # chunkings by design and are compared within themselves, not here.
    full_arms = {f"{prefix}_{s}" for s in ("mcq", "np", "mcq_np", "noimg", "easy")}
    for stem, recs in builds.items():
        if stem in full_arms:
            ref = ref_full
        elif removed_family(recs) and not re.fullmatch(rf"{prefix}_wh\d+", stem):
            # mcqnone40 / prose40 pair with wh40; ra_mcq_none pairs with the one wh file
            m = re.search(r"(\d+)$", stem)
            ref = (f"{prefix}_wh{m.group(1)}" if m and f"{prefix}_wh{m.group(1)}" in builds
                   else (whs[0] if len(whs) == 1 else None))
        else:
            ref = None
        if ref is None or stem == ref:
            n_skip += stem != ref_full and stem not in whs
            continue
        a, b = views[stem], views[ref]
        shared = [g for g in a if g in b]
        same_q = sum(1 for g in shared if a[g][0] == b[g][0])
        pooled = [g for g in shared if a[g][1] is not None and b[g][1] is not None]
        same_pool = sum(1 for g in pooled if a[g][1] == b[g][1])
        outside = len(a) - len(shared)
        bad = same_q < len(shared) or outside
        n_bad_align += bool(bad)
        if bad:
            print(f"  MISALIGNED {stem:22s} vs {ref:14s} groups {len(a):5d}, "
                  f"{outside} not in {ref}, {len(shared) - same_q} of {len(shared)} shared "
                  f"ids hold other questions")
        elif pooled and same_pool < len(pooled) and stem.endswith("_np"):
            print(f"  pool order {stem:22s} vs {ref:14s} same questions, "
                  f"{len(pooled) - same_pool} of {len(pooled)} pools in another order")

    n_files = n_stale_files = 0
    for stem, recs in builds.items():
        cur = {(r.get("item_id") or r["group_id"]): r for r in recs}
        cur_h = {k: phash(r["prompt"]) for k, r in cur.items()}
        for rp in sorted(glob.glob(os.path.join(d, f"{stem}.*.results.jsonl"))):
            model = os.path.basename(rp)[len(stem) + 1:-len(".results.jsonl")]
            n_files += 1
            stale = orphan = legacy_diff = 0
            res = load(rp)
            for r in res:
                k = r.get("key")
                if k not in cur:
                    orphan += 1
                elif r.get("phash"):
                    stale += r["phash"] != cur_h[k]
                elif [qnorm(q) for q in r.get("questions", [])] != \
                        [qnorm(q) for q in cur[k].get("questions", [])]:
                    legacy_diff += 1
            if stale or orphan or legacy_diff:
                n_stale_files += 1
                print(f"  STALE      {stem}.{model}: {len(res)} answers, {stale} for another "
                      f"prompt, {legacy_diff} for other questions (no hash), "
                      f"{orphan} for ids not in the build")
    print(f"  summary: {n_bad_align} misaligned arms, {n_stale_files} of {n_files} "
          f"results files with stale answers ({n_skip} ablation builds not paired, "
          f"not compared)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="+", required=True)
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    for p in a.prefix:
        audit(p, a.dir)


if __name__ == "__main__":
    main()

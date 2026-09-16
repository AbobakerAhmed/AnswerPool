#!/usr/bin/env python
# ============================================================================
# check_data.py — the blocking checks, before any model call.
#
# Answers three questions that determine whether the zero-generation
# construction is available at all:
#   1. QuALITY: how many questions share an article? (need N>=3-4 to be common)
#   2. Answer uniqueness: do sibling questions ever share a gold answer, or does
#      one question's gold also plausibly answer a sibling? (breaks pooling)
#   3. LongBench v2: do contexts repeat across instances?
#
# Usage:  python check_data.py
#         python check_data.py --hf-cache /path/to/cache
# ============================================================================
import argparse
import hashlib
import os
from collections import Counter, defaultdict


def _norm(s):
    return " ".join(str(s).split()).strip().lower()


def _hash(s, n=400):
    return hashlib.sha1(_norm(s)[:n].encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- QuALITY ---
def check_quality(cache_dir=None):
    print("\n" + "=" * 72)
    print("QuALITY — questions per article")
    print("=" * 72)
    from datasets import load_dataset

    ds = None
    for name, cfg in [("emozilla/quality", None), ("tasksource/quality", None)]:
        try:
            ds = load_dataset(name, cfg, split="validation", cache_dir=cache_dir)
            print(f"loaded: {name} (validation, {len(ds)} rows)")
            break
        except Exception as e:
            print(f"  (couldn't load {name}: {type(e).__name__})")
    if ds is None:
        print("!! could not load QuALITY — check HF name/availability")
        return

    cols = ds.column_names
    print(f"columns: {cols}")

    # find the article-ish and question-ish fields without assuming a schema
    art_col = next((c for c in ("article", "context", "passage", "text") if c in cols), None)
    q_col = next((c for c in ("question", "query") if c in cols), None)
    opt_col = next((c for c in ("options", "choices") if c in cols), None)
    ans_col = next((c for c in ("gold_label", "answer", "label", "gold") if c in cols), None)
    print(f"using -> article:{art_col}  question:{q_col}  options:{opt_col}  answer:{ans_col}")
    if not (art_col and q_col):
        print("!! unexpected schema — inspect ds[0] manually"); print(ds[0]); return

    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[_hash(row[art_col])].append(i)

    sizes = Counter(len(v) for v in groups.values())
    print(f"\narticles: {len(groups)} | questions: {len(ds)}")
    print("questions-per-article distribution:")
    for size in sorted(sizes):
        print(f"   {size:3d} q/article : {sizes[size]:5d} articles")

    for N in (3, 4, 5, 6):
        usable = sum(len(v) // N for v in groups.values() if len(v) >= N)
        print(f"   -> N={N}: {usable} full groups available")

    # ---- answer uniqueness within groups (the constraint that breaks pooling)
    if opt_col and ans_col:
        clashes = tot = 0
        for idxs in groups.values():
            if len(idxs) < 3:
                continue
            golds = []
            for i in idxs:
                row = ds[i]
                try:
                    a = row[ans_col]
                    opts = row[opt_col]
                    gold = opts[int(a) - 1] if isinstance(a, int) else (
                        opts[ord(str(a).upper()[0]) - 65] if str(a).upper()[0] in "ABCD" else a)
                    golds.append(_norm(gold))
                except Exception:
                    continue
            tot += 1
            if len(golds) != len(set(golds)):
                clashes += 1
        if tot:
            print(f"\nanswer-uniqueness: {clashes}/{tot} groups have a DUPLICATE gold "
                  f"({100*clashes/tot:.1f}%) -> these must be dropped or repaired")
    else:
        print("\n(skipped answer-uniqueness: couldn't identify options/answer columns)")


# ----------------------------------------------------------- LongBench v2 ---
def check_longbench_v2(cache_dir=None):
    print("\n" + "=" * 72)
    print("LongBench v2 — do contexts repeat across instances?")
    print("=" * 72)
    from datasets import load_dataset
    try:
        ds = load_dataset("THUDM/LongBench-v2", split="train", cache_dir=cache_dir)
    except Exception as e:
        print(f"!! could not load ({type(e).__name__}: {e})"); return
    print(f"rows: {len(ds)} | columns: {ds.column_names}")

    ctx = next((c for c in ("context", "passage", "text") if c in ds.column_names), None)
    if not ctx:
        print("!! no context column"); return

    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[_hash(row[ctx], 600)].append(i)
    sizes = Counter(len(v) for v in groups.values())
    print(f"distinct contexts: {len(groups)}")
    print("questions-per-context:")
    for size in sorted(sizes):
        print(f"   {size:3d} q/context : {sizes[size]:5d} contexts")
    reusable = sum(1 for v in groups.values() if len(v) >= 3)
    print(f"\n-> contexts with >=3 questions: {reusable}")
    print("   if ~0, the zero-generation conversion is NOT available here;"
          "\n   fall back to the claim->passage variant.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-cache", default=os.environ.get("HF_HOME"))
    ap.add_argument("--skip-longbench", action="store_true")
    a = ap.parse_args()
    check_quality(a.hf_cache)
    if not a.skip_longbench:
        check_longbench_v2(a.hf_cache)
    print("\ndone.")
